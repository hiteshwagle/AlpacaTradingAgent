from datetime import datetime, timedelta, timezone
import os
import unittest
from unittest.mock import Mock, patch

import requests

from integrations.tradingagents_api import (
    TradingAgentsAPIClient,
    TradingAgentsAPIError,
    build_trade_intent,
    map_analysts,
    reconcile_action,
    result_to_webui_state,
    validate_result_freshness,
)


def _result(*, rating="Buy", action="Buy", completed_at=None):
    return {
        "schema_version": "1.0",
        "symbol": "AAPL",
        "trade_date": "2026-09-26",
        "recommendation": {
            "rating": rating,
            "action": action,
            "entry_price": 250.0,
            "stop_loss": 240.0,
            "price_target": 275.0,
            "position_sizing": "5% of portfolio",
            "time_horizon": "3 months",
        },
        "final_decision": "Risk-adjusted decision.",
        "investment_plan": "Investment plan.",
        "trader_plan": "Trader plan.",
        "reports": {"news": "News report.", "macro": "Dedicated macro report."},
        "debates": {
            "investment_bull": "Bull case.",
            "investment_bear": "Bear case.",
            "risk_aggressive": "Aggressive case.",
            "risk_conservative": "Conservative case.",
            "risk_neutral": "Neutral case.",
        },
        "_api_metadata": {
            "analysis_id": "analysis-1",
            "completed_at": completed_at or datetime.now(timezone.utc).isoformat(),
        },
    }


class FakeResponse:
    def __init__(self, body, status_code=200):
        self.body = body
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests.HTTPError(f"HTTP {self.status_code}")
            error.response = self
            raise error

    def json(self):
        return self.body


class TradingAgentsAPITests(unittest.TestCase):
    def test_client_submits_authenticates_polls_and_validates_result(self):
        completed = _result()
        completed.pop("_api_metadata")
        session = Mock()
        session.request.side_effect = [
            FakeResponse({"analysis_id": "analysis-1", "status": "queued"}),
            FakeResponse(
                {
                    "analysis_id": "analysis-1",
                    "status": "completed",
                    "updated_at": "2026-09-26T01:02:03Z",
                    "result": completed,
                }
            ),
        ]
        client = TradingAgentsAPIClient(
            base_url="http://127.0.0.1:8000",
            api_key="test-key",
            session=session,
            sleeper=lambda _: None,
        )
        payload = {"symbol": "AAPL", "trade_date": "2026-09-26"}

        result = client.analyze(payload)

        self.assertEqual(result["_api_metadata"]["analysis_id"], "analysis-1")
        first_call = session.request.call_args_list[0]
        self.assertEqual(first_call.args[:2], ("POST", "http://127.0.0.1:8000/v1/analyses"))
        self.assertEqual(first_call.kwargs["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(session.request.call_args_list[1].args[0], "GET")

    def test_client_fails_closed_on_symbol_mismatch(self):
        session = Mock()
        result = _result()
        result.pop("_api_metadata")
        result["symbol"] = "MSFT"
        session.request.return_value = FakeResponse(
            {
                "analysis_id": "analysis-1",
                "status": "completed",
                "updated_at": "2026-09-26T01:02:03Z",
                "result": result,
            }
        )
        client = TradingAgentsAPIClient(session=session)
        with self.assertRaisesRegex(TradingAgentsAPIError, "symbol"):
            client.analyze({"symbol": "AAPL", "trade_date": "2026-09-26"})

    def test_macro_maps_to_dedicated_api_analyst(self):
        self.assertEqual(map_analysts(["market", "macro"]), ["market", "macro"])
        state = result_to_webui_state(_result(), ["market", "macro"])
        self.assertEqual(state["macro_report"], "Dedicated macro report.")
        self.assertEqual(state["investment_debate_state"]["bull_history"], "Bull case.")

    def test_reconcile_action_requires_directional_agreement(self):
        cases = [
            ("Buy", "Buy", "BUY"),
            ("Buy", "Hold", "HOLD"),
            ("Hold", "Buy", None),
            ("Sell", "Sell", "SELL"),
        ]
        for rating, action, expected in cases:
            with self.subTest(rating=rating, action=action):
                self.assertEqual(reconcile_action(_result(rating=rating, action=action)), expected)

    def test_build_trade_intent_uses_live_position_and_remote_risk_levels(self):
        intent = build_trade_intent(
            _result(),
            symbol="AAPL",
            current_position="NEUTRAL",
            allow_shorts=False,
            trade_date="2026-09-26",
        )
        self.assertEqual(intent["action"], "BUY")
        self.assertEqual(intent["current_position"], "NEUTRAL")
        self.assertEqual(intent["target_position"], "LONG")
        self.assertEqual(intent["risk_controls"]["stop_loss_price"], 240.0)
        self.assertEqual(intent["risk_controls"]["take_profit_price"], 275.0)

    def test_stale_analysis_is_rejected_before_execution(self):
        stale = datetime.now(timezone.utc) - timedelta(minutes=2)
        with patch.dict(os.environ, {"TRADINGAGENTS_API_MAX_ANALYSIS_AGE_SECONDS": "60"}):
            with self.assertRaisesRegex(TradingAgentsAPIError, "freshness"):
                validate_result_freshness(_result(completed_at=stale.isoformat()))


if __name__ == "__main__":
    unittest.main()
