"""Validated configuration and pure portfolio allocation rules."""
from __future__ import annotations

import math
import re
from datetime import date, datetime, timedelta, timezone
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator


def number(value, *, minimum=0.0):
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError("Invalid or negative numeric data")
    return result


def symbol(value):
    value = str(value).strip().upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,14}", value):
        raise ValueError("Expected a US equity symbol")
    return value


def utcnow():
    return datetime.now(timezone.utc)


def timestamp(value):
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timestamp must include a timezone")
    return result.astimezone(timezone.utc)


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    selected_symbols: list[str] = Field(default_factory=list, max_length=50)
    investment_fraction: float = Field(default=.8, gt=0, le=.8)
    cash_reserve_fraction: float = Field(default=.2, ge=.2, le=1)
    max_position_fraction: float = Field(default=.2, gt=0, le=.8)
    max_turnover_fraction: float = Field(default=.8, gt=0, le=2)
    max_daily_loss_fraction: float = Field(default=.05, gt=0, le=.2)
    max_spread_fraction: float = Field(default=.01, gt=0, le=.05)
    max_quote_age_seconds: int = Field(default=120, ge=5, le=300)
    max_research_age_seconds: int = Field(default=1800, ge=60, le=7200)
    min_trade_dollars: float = Field(default=10, ge=1, le=1000)
    slippage_buffer_fraction: float = Field(default=.005, ge=.001, le=.05)
    max_symbols: int = Field(default=75, ge=1, le=150)
    use_screener_regime: bool = True
    include_stock_screener_candidates: bool = True
    include_alpaca_activity_candidates: bool = True
    stock_screener_candidate_limit: int = Field(default=5, ge=1, le=10)
    alpaca_activity_candidate_limit: int = Field(default=5, ge=1, le=10)
    stock_screener_min_score: float = Field(default=70, ge=0, le=100)
    alpaca_activity_min_score: float = Field(default=60, ge=0, le=100)
    max_activity_scan_age_seconds: int = Field(default=300, ge=30, le=900)
    schedule_mode: str = "manual"
    interval_minutes: int = Field(default=60, ge=15, le=10080)
    daily_time: str = "10:00"
    timezone: str = "America/New_York"
    schedule_enabled: bool = False
    execution_enabled: bool = True
    learning_enabled: bool = True

    @field_validator("selected_symbols")
    @classmethod
    def symbols(cls, values):
        return sorted({symbol(v) for v in values})

    @field_validator("schedule_mode")
    @classmethod
    def mode(cls, value):
        if value not in {"manual", "daily", "interval"}:
            raise ValueError("Schedule must be manual, daily or interval")
        return value

    @field_validator("timezone")
    @classmethod
    def zone(cls, value):
        ZoneInfo(value)
        return value

    @field_validator("daily_time")
    @classmethod
    def daily(cls, value):
        if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value):
            raise ValueError("Use HH:MM, 24-hour time")
        return value


class HistoricalValidationRequest(BaseModel):
    """A bounded, point-in-time validation grid for the dashboard."""

    model_config = ConfigDict(extra="forbid")
    symbols: list[str] = Field(min_length=1, max_length=50)
    knowledge_cutoff: date
    analysis_dates: list[date] = Field(min_length=1, max_length=30)
    horizons: list[int] = Field(default_factory=lambda: [1, 5, 10, 20], min_length=1, max_length=8)
    x_posts_mode: Literal["disabled", "recent", "cache_only"] = "disabled"

    @field_validator("symbols")
    @classmethod
    def validation_symbols(cls, values):
        return sorted({symbol(value) for value in values})

    @field_validator("analysis_dates")
    @classmethod
    def dates(cls, values):
        result = sorted(set(values))
        if any(value >= utcnow().date() for value in result):
            raise ValueError("Analysis dates must be before today")
        return result

    @field_validator("horizons")
    @classmethod
    def outcome_horizons(cls, values):
        result = sorted(set(values))
        if any(value < 1 or value > 60 for value in result):
            raise ValueError("Horizons must be between 1 and 60 trading sessions")
        return result

    def model_post_init(self, __context):
        if any(value <= self.knowledge_cutoff for value in self.analysis_dates):
            raise ValueError("Every analysis date must be after the knowledge cutoff")
        if len(self.symbols) * len(self.analysis_dates) > 200:
            raise ValueError("Validation is limited to 200 symbol/date decisions per run")


def next_run(policy: Policy, now: datetime):
    if not policy.schedule_enabled or policy.schedule_mode == "manual":
        return None
    if policy.schedule_mode == "interval":
        return (now + timedelta(minutes=policy.interval_minutes)).isoformat()
    local = now.astimezone(ZoneInfo(policy.timezone))
    hour, minute = map(int, policy.daily_time.split(":"))
    candidate = local.replace(hour=hour, minute=minute, second=0, microsecond=0, fold=0)
    if candidate <= local:
        candidate += timedelta(days=1)
    # Round-trip resolves nonexistent local times during the spring DST change.
    return candidate.astimezone(timezone.utc).isoformat()


def allocate(account, positions, research, policy: Policy, *, exposure=1.0, learning=.999999):
    """Whole-book dollar targets. Existing HOLD positions participate in limits."""
    equity = number(account["equity"], minimum=.01)
    cash = number(account["cash"])
    previous = number(account["last_equity"], minimum=.01)
    if (previous - equity) / previous >= policy.max_daily_loss_fraction:
        raise ValueError("Daily loss limit reached; portfolio cycle blocked")
    if set(positions) - set(research):
        raise ValueError("Research missing for an existing position")
    deploy = min(policy.investment_fraction, 1 - policy.cash_reserve_fraction,
                 number(exposure), number(learning), 1.0)
    cap = equity * deploy
    per_position = equity * policy.max_position_fraction
    targets, weights = {}, {}
    for ticker, result in research.items():
        rec = result["recommendation"]
        action = str(rec.get("action", "")).upper()
        rating = str(rec.get("rating", "")).lower()
        compatible = {"buy": {"BUY", "HOLD"}, "overweight": {"BUY", "HOLD"},
                      "hold": {"HOLD"}, "underweight": {"SELL", "HOLD"}, "sell": {"SELL", "HOLD"}}
        if action not in compatible.get(rating, set()):
            raise ValueError(f"Conflicting research for {ticker}")
        held = number(positions.get(ticker, {}).get("market_value", 0))
        targets[ticker] = min(held, per_position) if action == "HOLD" else 0.0
        if action == "BUY":
            weights[ticker] = 2.0 if rating == "buy" else 1.0
    held_total = sum(targets.values())
    if held_total > cap:
        targets = {s: v * cap / held_total for s, v in targets.items()}
    remaining = max(0.0, cap - sum(targets.values()))
    active = dict(weights)
    while active and remaining > .01:
        total_weight = sum(active.values())
        allocated = 0.0
        for ticker, weight in list(active.items()):
            amount = min(remaining * weight / total_weight, per_position - targets[ticker])
            targets[ticker] += amount
            allocated += amount
            if targets[ticker] >= per_position - .01:
                del active[ticker]
        if allocated <= .01:
            break
        remaining -= allocated
    targets = {s: math.floor(v * 100) / 100 for s, v in targets.items()}
    turnover = sum(abs(v - number(positions.get(s, {}).get("market_value", 0))) for s, v in targets.items())
    if turnover > equity * policy.max_turnover_fraction + .01:
        raise ValueError("Proposed turnover exceeds the configured cycle limit")
    return {"targets": targets, "equity": equity, "cash": cash,
            "deployment_fraction": deploy, "minimum_cash": equity * (1 - deploy),
            "turnover": turnover, "policy_version": "portfolio-v1"}
