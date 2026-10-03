"""Bounded HTTP adapters. Broker writes have no automatic retry."""
from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta, timezone
from urllib.parse import quote, urlsplit

import certifi
import requests

from .models import number, symbol, timestamp


class RemoteError(RuntimeError):
    pass


def env_seconds(name, default, minimum, maximum):
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        raise ValueError(f"{name} must be numeric") from None
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} is outside the supported range")
    return value


class HTTPClient:
    def __init__(self, base, headers=None, session=None):
        parsed = urlsplit(base)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Invalid service URL")
        self.base = base.rstrip("/")
        self.headers = headers or {}
        self.session = session or requests.Session()

    def request(self, method, path, *, missing_ok=False, **kwargs):
        try:
            request_timeout = env_seconds(
                "TRADINGAGENTS_API_REQUEST_TIMEOUT_SECONDS", 30, 1, 120
            )
            response = self.session.request(method, self.base + path, headers=self.headers,
                                            timeout=(5, request_timeout), verify=certifi.where(),
                                            allow_redirects=False, **kwargs)
            if missing_ok and response.status_code == 404:
                return None
            if not 200 <= response.status_code < 300:
                raise RemoteError(f"Service rejected request (HTTP {response.status_code})")
            if response.status_code == 204:
                return {}
            return response.json()
        except (requests.RequestException, ValueError):
            raise RemoteError("Service unavailable or returned invalid JSON") from None


class PaperBroker(HTTPClient):
    def __init__(self, session=None):
        if os.getenv("ALPACA_USE_PAPER", "true").lower() not in {"true", "1", "yes"}:
            raise ValueError("Portfolio execution requires ALPACA_USE_PAPER=true")
        key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
        secret = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
        if not key or not secret:
            raise ValueError("Paper Alpaca credentials are missing")
        headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        # The endpoint is deliberately not configurable; live credentials cannot
        # make this worker place live orders.
        super().__init__("https://paper-api.alpaca.markets", headers, session)
        self.data = HTTPClient("https://data.alpaca.markets", headers, session)
        self.feed = os.getenv("ALPACA_DATA_FEED", "iex")
        if self.feed not in {"iex", "sip"}:
            raise ValueError("Execution requires an explicit iex or sip feed")

    def account(self):
        account = self.request("GET", "/v2/account")
        if account.get("status") != "ACTIVE" or any(account.get(k) for k in ("trading_blocked", "account_blocked", "trade_suspended_by_user")):
            raise ValueError("Broker account is not available for trading")
        if account.get("currency") != "USD" or not account.get("id"):
            raise ValueError("Expected a USD paper account")
        return account

    def positions(self):
        return self.request("GET", "/v2/positions")

    def clock(self):
        return self.request("GET", "/v2/clock")

    def open_orders(self):
        return self.request("GET", "/v2/orders", params={"status": "open", "limit": 500})

    def asset(self, ticker):
        asset = self.request("GET", "/v2/assets/" + quote(symbol(ticker), safe=""))
        if asset.get("class") != "us_equity" or not asset.get("tradable") or asset.get("status") != "active":
            raise ValueError(f"{ticker} is not a tradable US equity")
        return asset

    def quote(self, ticker, policy, now):
        body = self.data.request("GET", f"/v2/stocks/{quote(symbol(ticker), safe='')}/quotes/latest", params={"feed": self.feed})
        data = body["quote"]
        age = (now - timestamp(data["t"])).total_seconds()
        bid, ask = number(data["bp"], minimum=.0001), number(data["ap"], minimum=.0001)
        if age < -5 or age > policy.max_quote_age_seconds or ask < bid:
            raise ValueError(f"Stale or invalid quote for {ticker}")
        if (ask - bid) / ((ask + bid) / 2) > policy.max_spread_fraction:
            raise ValueError(f"Quote spread exceeds limit for {ticker}")
        return {"bid": bid, "ask": ask, "timestamp": data["t"], "feed": self.feed}

    def daily_bars(self, ticker, start_date, sessions=65):
        """Return adjusted daily bars after a historical decision date."""
        start = date.fromisoformat(str(start_date)) + timedelta(days=1)
        end = start + timedelta(days=max(30, int(sessions) * 3))
        body = self.data.request(
            "GET", f"/v2/stocks/{quote(symbol(ticker), safe='')}/bars",
            params={"timeframe": "1Day", "start": start.isoformat(), "end": end.isoformat(),
                    "limit": 1000, "adjustment": "all", "feed": self.feed},
        )
        rows = body.get("bars") if isinstance(body, dict) else None
        if not isinstance(rows, list):
            raise RemoteError("Historical market data returned an invalid response")
        result = []
        for row in rows:
            try:
                result.append({"date": str(row["t"])[:10], "open": number(row["o"], minimum=.0001)})
            except (KeyError, TypeError, ValueError):
                continue
        return sorted(result, key=lambda row: row["date"])

    def submit(self, payload):
        return self.request("POST", "/v2/orders", json=payload)

    def lookup(self, client_id):
        return self.request("GET", "/v2/orders:by_client_order_id", missing_ok=True,
                            params={"client_order_id": client_id})

    def cancel(self, order_id):
        return self.request("DELETE", f"/v2/orders/{quote(order_id, safe='')}")


class ResearchClient(HTTPClient):
    def __init__(self, session=None):
        api_key = os.getenv("TRADINGAGENTS_API_KEY", "")
        headers = {"Authorization": "Bearer " + api_key} if api_key else {}
        super().__init__(os.getenv("TRADINGAGENTS_API_URL", "http://127.0.0.1:8000"),
                         headers, session)

    def analyze(self, payload, on_job, stopped=lambda: False):
        job = self.request("POST", "/v1/analyses", json=payload)
        job_id = job["analysis_id"]
        on_job(job_id)
        deadline = time.monotonic() + env_seconds(
            "TRADINGAGENTS_API_TIMEOUT_SECONDS", 900, 60, 7200
        )
        poll_seconds = env_seconds("TRADINGAGENTS_API_POLL_SECONDS", 2, .25, 30)
        while job.get("status") not in {"completed", "failed", "cancelled"}:
            if stopped() or time.monotonic() > deadline:
                self.request("POST", f"/v1/analyses/{quote(job_id, safe='')}/cancel")
                raise RemoteError("Research cancelled or timed out")
            time.sleep(poll_seconds)
            job = self.request("GET", f"/v1/analyses/{quote(job_id, safe='')}")
        if job["status"] != "completed":
            raise RemoteError("Research did not complete")
        result = job["result"]
        if (result.get("symbol") != payload["symbol"] or result.get("trade_date") != payload["trade_date"]
                or result.get("schema_version") != "1.0" or result.get("asset_type") != "stock"):
            raise RemoteError("Research result identity mismatch")
        return {**result, "completed_at": job["updated_at"], "analysis_id": job_id}

    def scan(self, top_n=20):
        if not isinstance(top_n, int) or not 1 <= top_n <= 50:
            raise ValueError("Activity scanner limit must be between 1 and 50")
        result = self.request("POST", "/v1/scanner/scan", json={"top_n": top_n})
        if (not isinstance(result, dict) or result.get("schema_version") != "1.0"
                or result.get("market") != "US"
                or not isinstance(result.get("candidates"), list)):
            raise RemoteError("Unsupported Alpaca activity scanner response")
        return result


class ScreenerClient(HTTPClient):
    # Fixed route mapping prevents the dashboard from becoming an arbitrary proxy.
    RESOURCES = {"daily": "/market-scan/daily-snapshot", "breadth": "/breadth/current",
                 "groups": "/groups/rankings/current", "rrg": "/groups/rrg",
                 "scans": "/scans", "themes": "/themes", "watchlists": "/user-watchlists",
                 "validation": "/validation/overview", "digest": "/digest/daily",
                 "options": "/options-analytics/command-center"}

    def __init__(self, session=None):
        super().__init__(os.getenv("STOCK_SCREENER_API_URL", "http://127.0.0.1:8080/api/v1"),
                         {"X-Server-Auth": os.getenv("STOCK_SCREENER_PASSWORD", "")}, session)

    def read(self, resource, ticker=None):
        if resource == "symbol":
            path = "/stocks/" + quote(symbol(ticker), safe="") + "/decision-dashboard"
        else:
            path = self.RESOURCES[resource]
        return self.request("GET", path, params={"market": "US"})

    def daily(self):
        result = self.read("daily")
        if result.get("market") != "US" or result.get("schema_version") != 1:
            raise RemoteError("Unsupported stock-screener daily snapshot")
        return result

    def scan(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("Scan request must be an object")
        request = dict(payload)
        if request.pop("market", "US") != "US":
            raise ValueError("Only US scans are supported")
        universe = request.get("universe_def")
        if universe is None:
            request["universe_def"] = {"type": "market", "market": "US"}
        elif not isinstance(universe, dict) or universe.get("type") != "market" or universe.get("market") != "US":
            raise ValueError("Portfolio scans require a US market universe")
        request.pop("universe", None)
        return self.request("POST", "/scans", json=request)

    def scan_results(self, scan_id):
        return self.request("GET", "/scans/" + quote(scan_id, safe="") + "/results")
