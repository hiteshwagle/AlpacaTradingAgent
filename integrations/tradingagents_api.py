"""Fail-closed client and adapter for the local TradingAgents API."""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Callable, Iterable, Optional

import requests
from dotenv import load_dotenv

from tradingagents.agents.schemas import (
    AdvisoryRating,
    ExecutableAction,
    RiskDecision,
    build_trade_intent_from_risk_decision,
)

load_dotenv()

SUPPORTED_ANALYSTS = {"market", "social", "news", "fundamentals", "macro"}
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


class TradingAgentsAPIError(RuntimeError):
    """Raised when remote research cannot be safely consumed."""


def _positive_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise TradingAgentsAPIError(f"{name} must be a number.") from exc
    if value <= 0:
        raise TradingAgentsAPIError(f"{name} must be greater than zero.")
    return value


def _symbol_key(value: object) -> str:
    return "".join(character for character in str(value or "").upper() if character.isalnum())


class TradingAgentsAPIClient:
    """Submit and poll TradingAgents analyses without a local graph fallback."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        request_timeout: Optional[float] = None,
        analysis_timeout: Optional[float] = None,
        poll_interval: Optional[float] = None,
        session: Optional[requests.Session] = None,
        sleeper: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.base_url = (base_url or os.getenv("TRADINGAGENTS_API_URL", "http://127.0.0.1:8000")).rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("TRADINGAGENTS_API_KEY", "").strip()
        self.request_timeout = request_timeout or _positive_float("TRADINGAGENTS_API_REQUEST_TIMEOUT_SECONDS", 30.0)
        self.analysis_timeout = analysis_timeout or _positive_float("TRADINGAGENTS_API_TIMEOUT_SECONDS", 900.0)
        self.poll_interval = poll_interval or _positive_float("TRADINGAGENTS_API_POLL_SECONDS", 2.0)
        self.session = session or requests.Session()
        self._sleep = sleeper
        self._monotonic = monotonic

    @property
    def headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request(self, method: str, path: str, **kwargs) -> dict:
        try:
            response = self.session.request(
                method,
                f"{self.base_url}{path}",
                headers=self.headers,
                timeout=self.request_timeout,
                **kwargs,
            )
            response.raise_for_status()
            body = response.json()
        except requests.RequestException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            suffix = f" (HTTP {status})" if status else ""
            raise TradingAgentsAPIError(f"TradingAgents API request failed{suffix}: {exc}") from exc
        except ValueError as exc:
            raise TradingAgentsAPIError("TradingAgents API returned invalid JSON.") from exc
        if not isinstance(body, dict):
            raise TradingAgentsAPIError("TradingAgents API returned an invalid response object.")
        return body

    def submit_analysis(self, payload: dict) -> dict:
        job = self._request("POST", "/v1/analyses", json=payload)
        if not job.get("analysis_id"):
            raise TradingAgentsAPIError("TradingAgents API did not return an analysis id.")
        return job

    def get_analysis(self, analysis_id: str) -> dict:
        return self._request("GET", f"/v1/analyses/{analysis_id}")

    def cancel_analysis(self, analysis_id: str) -> None:
        try:
            self._request("POST", f"/v1/analyses/{analysis_id}/cancel")
        except TradingAgentsAPIError:
            pass

    def analyze(
        self,
        payload: dict,
        *,
        on_status: Optional[Callable[[dict], None]] = None,
    ) -> dict:
        job = self.submit_analysis(payload)
        analysis_id = str(job["analysis_id"])
        deadline = self._monotonic() + self.analysis_timeout
        last_status = None

        while True:
            status = str(job.get("status", "")).lower()
            if on_status is not None and status != last_status:
                on_status(job)
                last_status = status

            if status in TERMINAL_STATUSES:
                break
            if self._monotonic() >= deadline:
                self.cancel_analysis(analysis_id)
                raise TradingAgentsAPIError(
                    f"TradingAgents analysis {analysis_id} timed out after {self.analysis_timeout:g} seconds."
                )
            self._sleep(self.poll_interval)
            job = self.get_analysis(analysis_id)

        if status != "completed":
            error = job.get("error") or f"analysis ended with status {status or 'unknown'}"
            if isinstance(error, dict):
                error = error.get("message") or error.get("code") or "unknown API error"
            raise TradingAgentsAPIError(f"TradingAgents analysis {analysis_id} failed: {error}")

        result = job.get("result")
        if not isinstance(result, dict):
            raise TradingAgentsAPIError("Completed TradingAgents analysis has no result object.")
        self._validate_result(result, payload)
        result = dict(result)
        result["_api_metadata"] = {
            "analysis_id": analysis_id,
            "completed_at": job.get("updated_at"),
            "api_url": self.base_url,
        }
        return result

    @staticmethod
    def _validate_result(result: dict, payload: dict) -> None:
        if str(result.get("schema_version")) != "1.0":
            raise TradingAgentsAPIError("Unsupported TradingAgents result schema version.")
        if _symbol_key(result.get("symbol")) != _symbol_key(payload.get("symbol")):
            raise TradingAgentsAPIError("TradingAgents result symbol does not match the request.")
        if str(result.get("trade_date")) != str(payload.get("trade_date")):
            raise TradingAgentsAPIError("TradingAgents result date does not match the request.")
        if not isinstance(result.get("recommendation"), dict):
            raise TradingAgentsAPIError("TradingAgents result has no structured recommendation.")


def map_analysts(selected_analysts: Iterable[str]) -> list[str]:
    """Validate and order analyst selections for the TradingAgents API."""
    selected = {str(item).lower() for item in selected_analysts}
    mapped = [
        name
        for name in ("market", "social", "news", "fundamentals", "macro")
        if name in selected
    ]
    if not mapped:
        raise TradingAgentsAPIError("At least one supported TradingAgents analyst must be selected.")
    return mapped


def reconcile_action(result: dict) -> Optional[str]:
    """Accept only a trader action compatible with the final advisory rating."""
    recommendation = result.get("recommendation") or {}
    rating = str(recommendation.get("rating") or "").strip().lower()
    action = str(recommendation.get("action") or "").strip().upper()
    compatible = {
        "buy": {"BUY", "HOLD"},
        "overweight": {"BUY", "HOLD"},
        "hold": {"HOLD"},
        "underweight": {"SELL", "HOLD"},
        "sell": {"SELL", "HOLD"},
    }
    return action if action in compatible.get(rating, set()) else None


def validate_result_freshness(result: dict, *, now: Optional[datetime] = None) -> None:
    completed_at = (result.get("_api_metadata") or {}).get("completed_at")
    if not completed_at:
        raise TradingAgentsAPIError("TradingAgents result has no completion timestamp; trade blocked.")
    try:
        completed = datetime.fromisoformat(str(completed_at).replace("Z", "+00:00"))
    except ValueError as exc:
        raise TradingAgentsAPIError("TradingAgents completion timestamp is invalid; trade blocked.") from exc
    if completed.tzinfo is None:
        completed = completed.replace(tzinfo=timezone.utc)
    current = now or datetime.now(timezone.utc)
    age = (current.astimezone(timezone.utc) - completed.astimezone(timezone.utc)).total_seconds()
    max_age = _positive_float("TRADINGAGENTS_API_MAX_ANALYSIS_AGE_SECONDS", 1800.0)
    if age < -60 or age > max_age:
        raise TradingAgentsAPIError(
            f"TradingAgents result is outside the allowed freshness window ({max_age:g} seconds); trade blocked."
        )


def _optional_text(value: object) -> Optional[str]:
    if value in (None, ""):
        return None
    return str(value)


def build_trade_intent(
    result: dict,
    *,
    symbol: str,
    current_position: str,
    allow_shorts: bool,
    trade_date: str,
) -> dict:
    """Convert accepted remote research into Alpaca's deterministic contract."""
    validate_result_freshness(result)
    action = reconcile_action(result)
    if action is None:
        raise TradingAgentsAPIError(
            "TradingAgents rating and trader action are missing or inconsistent; trade blocked."
        )
    recommendation = result["recommendation"]
    rating = AdvisoryRating(str(recommendation["rating"]).title())
    decision = RiskDecision(
        action=ExecutableAction(action),
        confidence="unspecified by TradingAgents API",
        risk_rationale=str(result.get("final_decision") or "No rationale supplied."),
        required_controls=(
            "Alpaca must independently verify the live position, account constraints, "
            "portfolio exposure, sizing, and market rules before execution."
        ),
        advisory_rating=rating,
        entry_guidance=_optional_text(recommendation.get("entry_price")),
        stop_loss=_optional_text(recommendation.get("stop_loss")),
        take_profit=_optional_text(recommendation.get("price_target")),
        invalidation=None,
        max_position_size=_optional_text(recommendation.get("position_sizing")),
        time_horizon=_optional_text(recommendation.get("time_horizon")),
    )
    intent = build_trade_intent_from_risk_decision(
        symbol=symbol,
        trading_mode="investment",
        current_position=current_position,
        decision=decision,
        allow_shorts=allow_shorts,
        trade_date=trade_date,
    )
    return intent.model_dump(mode="json")


def result_to_webui_state(result: dict, selected_analysts: Iterable[str]) -> dict:
    reports = result.get("reports") or {}
    debates = result.get("debates") or {}
    recommendation = result.get("recommendation") or {}
    selected = {str(item).lower() for item in selected_analysts}
    news = str(reports.get("news") or "")
    macro = str(reports.get("macro") or "") if "macro" in selected else ""

    investment_history = str(debates.get("investment") or "")
    bull = str(debates.get("investment_bull") or "")
    bear = str(debates.get("investment_bear") or "")
    investment_judgement = str(debates.get("investment_judgement") or "")
    risk_history = str(debates.get("risk") or "")
    risky = str(debates.get("risk_aggressive") or "")
    safe = str(debates.get("risk_conservative") or "")
    neutral = str(debates.get("risk_neutral") or "")
    risk_judgement = str(debates.get("risk_judgement") or result.get("final_decision") or "")
    trader_plan = str(result.get("trader_plan") or "")

    return {
        "company_of_interest": result.get("symbol"),
        "trade_date": result.get("trade_date"),
        "trading_mode": "investment",
        "market_report": str(reports.get("market") or ""),
        "sentiment_report": str(reports.get("sentiment") or ""),
        "news_report": news,
        "fundamentals_report": str(reports.get("fundamentals") or ""),
        "macro_report": macro,
        "investment_debate_state": {
            "bull_history": bull,
            "bear_history": bear,
            "history": investment_history,
            "judge_decision": investment_judgement,
            "bull_messages": [bull] if bull else [],
            "bear_messages": [bear] if bear else [],
        },
        "investment_plan": str(result.get("investment_plan") or investment_judgement),
        "trader_investment_plan": trader_plan,
        "risk_debate_state": {
            "risky_history": risky,
            "safe_history": safe,
            "neutral_history": neutral,
            "history": risk_history,
            "judge_decision": risk_judgement,
            "risky_messages": [risky] if risky else [],
            "safe_messages": [safe] if safe else [],
            "neutral_messages": [neutral] if neutral else [],
        },
        "final_trade_decision": str(result.get("final_decision") or ""),
        "final_trade_intent": None,
        "api_result": result,
    }
