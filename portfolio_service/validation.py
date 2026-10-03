"""Persistent walk-forward historical validation for the simple dashboard."""
from __future__ import annotations

from statistics import mean

from .models import HistoricalValidationRequest


def score_decision(action, bars, benchmark_bars, horizons):
    """Score one saved decision using next-session-open execution."""
    normalized = str(action or "").upper()
    if normalized not in {"BUY", "HOLD", "SELL"}:
        raise ValueError("Research did not return a scorable Buy, Hold or Sell action")
    if len(bars) < 2:
        raise ValueError("Not enough later market sessions to score this decision")
    entry = float(bars[0]["open"])
    benchmark_by_date = {row["date"]: float(row["open"]) for row in benchmark_bars}
    benchmark_entry = benchmark_by_date.get(bars[0]["date"])
    outcomes = []
    for horizon in horizons:
        if len(bars) <= horizon:
            outcomes.append({"horizon": horizon, "available": False})
            continue
        asset_return = float(bars[horizon]["open"]) / entry - 1
        benchmark_return = None
        benchmark_exit = benchmark_by_date.get(bars[horizon]["date"])
        if benchmark_entry and benchmark_exit:
            benchmark_return = benchmark_exit / benchmark_entry - 1
        decision_return = asset_return if normalized == "BUY" else -asset_return if normalized == "SELL" else 0.0
        correct = asset_return > 0 if normalized == "BUY" else asset_return < 0 if normalized == "SELL" else abs(asset_return) < .02
        outcomes.append({
            "horizon": horizon, "available": True, "exit_date": bars[horizon]["date"],
            "asset_return": asset_return, "decision_return": decision_return,
            "benchmark_return": benchmark_return,
            "alpha": asset_return - benchmark_return if benchmark_return is not None else None,
            "correct": correct,
        })
    return {"entry_date": bars[0]["date"], "entry_price": entry, "outcomes": outcomes}


def summarize(results):
    available = [outcome for row in results for outcome in row.get("score", {}).get("outcomes", [])
                 if outcome.get("available")]
    decisions = [row for row in results if row.get("status") == "completed"]
    return {
        "decisions_completed": len(decisions),
        "decisions_failed": len(results) - len(decisions),
        "outcomes_scored": len(available),
        "directional_accuracy": (
            sum(bool(row["correct"]) for row in available) / len(available) if available else None
        ),
        "mean_decision_return": mean(row["decision_return"] for row in available) if available else None,
        "mean_asset_alpha": mean(row["alpha"] for row in available if row.get("alpha") is not None)
        if any(row.get("alpha") is not None for row in available) else None,
    }


class HistoricalValidator:
    def __init__(self, store, broker, research):
        self.store = store
        self.broker = broker
        self.research = research

    def run(self, run_id):
        row = self.store.validation(run_id)
        if not row or row["status"] != "queued":
            return
        request = HistoricalValidationRequest.model_validate(row["request"])
        payload = row["payload"]
        total = len(request.symbols) * len(request.analysis_dates)
        completed = 0
        benchmark_cache = {}
        payload["progress"] = {"step": "analyzing", "message": "Starting historical validation",
                               "percent": 1, "completed": 0, "total": total}
        self.store.update_validation(run_id, "running", payload)
        try:
            for analysis_date in request.analysis_dates:
                date_text = analysis_date.isoformat()
                if date_text not in benchmark_cache:
                    try:
                        benchmark_cache[date_text] = self.broker.daily_bars(
                            "SPY", date_text, max(request.horizons) + 2
                        )
                    except Exception:
                        # Benchmark alpha is optional; individual symbol outcomes
                        # remain useful when SPY history is temporarily unavailable.
                        benchmark_cache[date_text] = []
                for ticker in request.symbols:
                    payload["progress"] = {
                        "step": "analyzing", "message": f"Analyzing {ticker} as of {date_text}",
                        "symbol": ticker, "analysis_date": date_text, "completed": completed,
                        "total": total, "percent": max(1, round(completed / total * 95)),
                    }
                    self.store.update_validation(run_id, "running", payload)
                    result_row = {"symbol": ticker, "analysis_date": date_text}
                    try:
                        result = self.research.analyze({
                            "symbol": ticker, "trade_date": date_text, "asset_type": "stock",
                            "analysts": ["market", "social", "news", "fundamentals", "macro"],
                            "options": {"output_language": "English", "save_reports": False,
                                        "x_posts_mode": request.x_posts_mode},
                        }, lambda job_id: result_row.update(analysis_id=job_id))
                        action = result.get("recommendation", {}).get("action")
                        bars = self.broker.daily_bars(ticker, date_text, max(request.horizons) + 2)
                        result_row.update(
                            status="completed", recommendation=result.get("recommendation", {}),
                            score=score_decision(action, bars, benchmark_cache[date_text], request.horizons),
                        )
                    except Exception as exc:
                        # Persist a safe error class only; remote bodies and credentials never enter the UI.
                        result_row.update(status="failed", error=type(exc).__name__)
                    payload["results"].append(result_row)
                    completed += 1
                    payload["progress"] = {
                        "step": "scoring", "message": f"Scored {ticker} for {date_text}",
                        "symbol": ticker, "analysis_date": date_text, "completed": completed,
                        "total": total, "percent": round(completed / total * 95),
                    }
                    self.store.update_validation(run_id, "running", payload)
            payload["summary"] = summarize(payload["results"])
            payload["progress"] = {"step": "completed", "message": "Historical validation complete",
                                   "completed": total, "total": total, "percent": 100}
            self.store.update_validation(run_id, "completed", payload)
        except Exception as exc:
            payload["error"] = type(exc).__name__
            payload["progress"] = {"step": "failed", "message": "Historical validation stopped",
                                   "completed": completed, "total": total,
                                   "percent": round(completed / total * 95)}
            self.store.update_validation(run_id, "failed", payload)
