"""Authenticated server-side portfolio API and embedded dashboard."""
import hmac
import os
from urllib.parse import urlsplit

from flask import Response, jsonify, redirect, render_template, request

from .clients import PaperBroker, ResearchClient, ScreenerClient
from .models import HistoricalValidationRequest, Policy
from .store import Store


def enabled():
    return os.getenv("PORTFOLIO_ENABLED", "false").lower() in {"true", "yes", "1"}


def register(server, store=None, broker_factory=PaperBroker):
    store = store or Store()
    user = os.getenv("DASHBOARD_USERNAME", "admin")
    password = os.getenv("DASHBOARD_PASSWORD", "")
    production = os.getenv("DEPLOYMENT_MODE") == "production"
    if production and len(password) < 16:
        raise ValueError("Production requires DASHBOARD_PASSWORD of at least 16 characters")
    server.config["MAX_CONTENT_LENGTH"] = 1024 * 1024

    @server.before_request
    def protect():
        if request.path == "/healthz":
            return None
        if password:
            auth = request.authorization
            if not auth or auth.type != "basic" or not hmac.compare_digest(auth.username or "", user) or not hmac.compare_digest(auth.password or "", password):
                return Response("Authentication required", 401, {"WWW-Authenticate": 'Basic realm="Trading dashboard"'})
        elif request.remote_addr not in {"127.0.0.1", "::1"}:
            return jsonify(error="Dashboard password is required for remote access"), 403
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("Origin")
            if request.headers.get("Sec-Fetch-Site") == "cross-site" or (origin and urlsplit(origin).netloc != request.host):
                return jsonify(error="Cross-origin mutation rejected"), 403
            if request.path.startswith("/api/portfolio/") and request.headers.get("X-Portfolio-Request") != "1":
                return jsonify(error="Missing request protection header"), 403

    @server.after_request
    def secure(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["Referrer-Policy"] = "same-origin"
        if request.path.startswith("/api/portfolio"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @server.get("/healthz")
    def health():
        return jsonify(status="ok")

    if enabled():
        @server.get("/")
        def home():
            return redirect("/portfolio")

        @server.get("/analysis")
        def analysis():
            return redirect("/research/")

    @server.get("/portfolio")
    def portfolio():
        return render_template("portfolio.html")

    @server.get("/api/portfolio/status")
    def status():
        return jsonify(policy=store.policy().model_dump(), halted=store.get("halted", False),
                       heartbeat=store.get("heartbeat"), next_run=store.get("next_run"),
                       worker_error=store.get("worker_error"), cycles=store.cycles(20),
                       orders=store.orders(), evaluations=store.evaluations(store.get("account_id", "")))

    @server.get("/api/portfolio/overview")
    def overview():
        try:
            broker = broker_factory()
            account = broker.account()
            positions = broker.positions()
            clock = broker.clock()

            def value(row, key, default=0):
                try:
                    return float(row.get(key, default))
                except (TypeError, ValueError):
                    return default

            holdings = [{
                "symbol": str(row.get("symbol", "")),
                "quantity": value(row, "qty"),
                "market_value": value(row, "market_value"),
                "average_price": value(row, "avg_entry_price"),
                "current_price": value(row, "current_price"),
                "profit_loss": value(row, "unrealized_pl"),
                "profit_loss_percent": value(row, "unrealized_plpc") * 100,
            } for row in positions if row.get("asset_class") == "us_equity"]
            holdings.sort(key=lambda row: row["market_value"], reverse=True)
            equity = value(account, "equity")
            previous = value(account, "last_equity")
            return jsonify(
                account={
                    "portfolio_value": equity,
                    "cash": value(account, "cash"),
                    "buying_power": value(account, "buying_power"),
                    "invested": sum(row["market_value"] for row in holdings),
                    "today_change": equity - previous,
                    "today_change_percent": ((equity / previous - 1) * 100) if previous else 0,
                },
                positions=holdings,
                market_open=bool(clock.get("is_open")),
                next_market_open=clock.get("next_open"),
                next_market_close=clock.get("next_close"),
                paper=True,
            )
        except Exception as exc:
            return jsonify(error="Account information is temporarily unavailable",
                           reason=type(exc).__name__), 503

    @server.post("/api/portfolio/policy")
    def policy():
        try:
            policy = Policy.model_validate(request.get_json())
            store.save_policy(policy)
            return jsonify(policy=policy.model_dump())
        except (ValueError, KeyError):
            return jsonify(error="Invalid policy: check symbols, schedule and risk bounds"), 400

    @server.post("/api/portfolio/run")
    def run():
        try:
            return jsonify(cycle_id=store.enqueue()), 202
        except ValueError as exc:
            return jsonify(error=str(exc)), 409

    @server.get("/api/portfolio/validations/latest")
    def latest_validation():
        rows = store.validations(1)
        return jsonify(validation=rows[0] if rows else None,
                       worker_heartbeat=store.get("validation_heartbeat"),
                       worker_error=store.get("validation_worker_error"))

    @server.post("/api/portfolio/validations")
    def create_validation():
        try:
            validation_request = HistoricalValidationRequest.model_validate(request.get_json())
            return jsonify(validation_id=store.enqueue_validation(validation_request)), 202
        except ValueError as exc:
            message = str(exc)
            if "already running" in message:
                return jsonify(error=message), 409
            return jsonify(error="Invalid historical validation settings"), 400

    @server.post("/api/portfolio/halt")
    def halt():
        value = (request.get_json() or {}).get("halted")
        if not isinstance(value, bool):
            return jsonify(error="halted must be a boolean"), 400
        store.set("halted", value)
        store.event(None, "emergency_stop", {"halted": value})
        return jsonify(halted=value)

    @server.get("/api/portfolio/discovery/<resource>")
    def discovery(resource):
        try:
            if resource == "alpaca":
                return jsonify(ResearchClient().scan())
            if resource not in {*ScreenerClient.RESOURCES, "symbol"}:
                return jsonify(error="Unknown discovery view"), 404
            return jsonify(ScreenerClient().read(resource, request.args.get("symbol")))
        except Exception as exc:
            return jsonify(error="Discovery unavailable", reason=type(exc).__name__), 503

    @server.post("/api/portfolio/scans")
    def scan():
        try:
            return jsonify(ScreenerClient().scan(request.get_json())), 202
        except ValueError:
            return jsonify(error="Invalid US scan request"), 400
        except Exception as exc:
            return jsonify(error="Scan could not start", reason=type(exc).__name__), 503

    @server.get("/api/portfolio/scans/<scan_id>/results")
    def scan_results(scan_id):
        try:
            return jsonify(ScreenerClient().scan_results(scan_id))
        except Exception as exc:
            return jsonify(error="Scan results unavailable", reason=type(exc).__name__), 503

    return store
