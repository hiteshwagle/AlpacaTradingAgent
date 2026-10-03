from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from portfolio_service.clients import (
    PaperBroker,
    RemoteError,
    ResearchClient,
    ScreenerClient,
)
from portfolio_service.engine import Engine
from portfolio_service.models import (
    HistoricalValidationRequest,
    Policy,
    allocate,
    next_run,
)
from portfolio_service.store import Store
from portfolio_service.validation import (
    HistoricalValidator,
    apply_outcome_matrix,
    outcome_verdict,
    score_decision,
    summarize,
)
from portfolio_service.web import register
from portfolio_service.worker import tick


def result(action, rating):
    return {"recommendation": {"action": action, "rating": rating}}


class FakeResponse:
    def __init__(self, status=200, body=None):
        self.status_code, self.body, self.headers = status, body, {}

    def json(self):
        return self.body


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.responses.pop(0)


class FakeBroker:
    def __init__(self, *, open_market=True, fill_status="filled", existing_orders=False,
                 untradable=()):
        self.open_market, self.fill_status = open_market, fill_status
        self.existing_orders = existing_orders
        self.untradable = set(untradable)
        self.submitted, self.responses = [], {}
        self.position_rows = {"OLD": {"symbol": "OLD", "asset_class": "us_equity",
                                      "qty": "100", "market_value": "1000",
                                      "avg_entry_price": "9"}}
        self.account_row = {"id": "paper-1", "status": "ACTIVE", "currency": "USD",
                            "cash": "9000", "equity": "10000", "last_equity": "10000",
                            "buying_power": "9000", "trading_blocked": False,
                            "account_blocked": False, "trade_suspended_by_user": False}

    def clock(self): return {"is_open": self.open_market}
    def account(self): return dict(self.account_row)
    def positions(self): return [dict(v) for v in self.position_rows.values()]
    def open_orders(self): return [{"id": "external"}] if self.existing_orders else []
    def asset(self, ticker):
        if ticker in self.untradable:
            raise ValueError("not tradable")
        return {"class": "us_equity", "status": "active", "tradable": True, "fractionable": True}
    def quote(self, ticker, policy, now): return {"bid": 10, "ask": 10, "timestamp": now.isoformat(), "feed": "iex"}
    def cancel(self, order_id): return {}

    def submit(self, request):
        self.submitted.append(dict(request))
        quantity = float(request.get("qty") or float(request["notional"]) / 10)
        current = float(self.position_rows.get(request["symbol"], {}).get("qty", 0))
        filled = quantity if self.fill_status == "filled" else quantity / 2
        new_qty = current + (filled if request["side"] == "buy" else -filled)
        if self.fill_status == "filled":
            if new_qty <= .000001:
                self.position_rows.pop(request["symbol"], None)
            else:
                self.position_rows[request["symbol"]] = {"symbol": request["symbol"],
                    "asset_class": "us_equity", "qty": str(new_qty),
                    "market_value": str(new_qty * 10), "avg_entry_price": "10"}
        response = {"id": request["client_order_id"], "client_order_id": request["client_order_id"],
                    "status": self.fill_status, "filled_qty": str(filled)}
        self.responses[request["client_order_id"]] = response
        return response

    def lookup(self, client_id): return self.responses.get(client_id)


class FakeResearch:
    def __init__(self, candidates=None, *, as_of="2026-10-01T15:00:00+00:00",
                 session="regular"):
        self.candidates = candidates or []
        self.as_of = as_of
        self.session = session
        self.analyzed = []

    def scan(self, top_n=20):
        return {"schema_version": "1.0", "market": "US", "session": self.session,
                "as_of": self.as_of, "feed": "iex", "metadata": {},
                "candidates": self.candidates[:top_n]}

    def analyze(self, payload, on_job, stopped):
        self.analyzed.append(payload["symbol"])
        on_job("job-" + payload["symbol"])
        action = "Sell" if payload["symbol"] == "OLD" else "Buy"
        return {"schema_version": "1.0", "symbol": payload["symbol"],
                "trade_date": payload["trade_date"], "asset_type": "stock",
                "recommendation": {"action": action, "rating": action},
                "completed_at": "2026-10-01T15:00:00+00:00"}


class FakeScreener:
    def __init__(self, candidates=None, leaders=None, *, snapshot_date="2026-10-01"):
        self.candidates = candidates or []
        self.leaders = leaders or []
        self.snapshot_date = snapshot_date

    def daily(self):
        return {"market": "US", "schema_version": 1,
                "freshness": {"snapshot_as_of_date": self.snapshot_date},
                "top_candidates": {"rows": self.candidates},
                "leaders": {"rows": self.leaders},
                "market_health_exposure": {"date": self.snapshot_date, "exposure_score": 80}}


class PolicyTests(unittest.TestCase):
    def test_defaults_match_requested_budget_and_cash(self):
        policy = Policy()
        self.assertEqual(policy.investment_fraction, .8)
        self.assertEqual(policy.cash_reserve_fraction, .2)
        self.assertTrue(policy.execution_enabled)
        self.assertTrue(policy.include_stock_screener_candidates)
        self.assertTrue(policy.include_alpaca_activity_candidates)
        self.assertEqual(policy.stock_screener_candidate_limit, 5)
        self.assertEqual(policy.alpaca_activity_candidate_limit, 5)

    def test_limits_cannot_remove_cash_floor_or_raise_budget(self):
        for value in ({"investment_fraction": .81}, {"cash_reserve_fraction": .19}):
            with self.assertRaises(ValueError):
                Policy.model_validate(value)

    def test_daily_dst_schedule_is_timezone_aware(self):
        policy = Policy(schedule_enabled=True, schedule_mode="daily", daily_time="10:00")
        due = next_run(policy, datetime(2026, 3, 8, 14, 30, tzinfo=timezone.utc))
        self.assertEqual(due, "2026-03-09T14:00:00+00:00")

    def test_allocation_includes_holdings_and_preserves_cash(self):
        account = {"equity": "10000", "cash": "5000", "last_equity": "10000"}
        positions = {"AAPL": {"market_value": "3000"}}
        plan = allocate(account, positions, {"AAPL": result("Hold", "Hold"),
                                             "MSFT": result("Buy", "Buy")}, Policy())
        self.assertLessEqual(sum(plan["targets"].values()), 8000)
        self.assertAlmostEqual(plan["minimum_cash"], 2000)
        self.assertLessEqual(plan["targets"]["AAPL"], 2000)  # concentration cap

    def test_missing_holding_research_and_conflicting_result_fail_closed(self):
        account = {"equity": 10000, "cash": 5000, "last_equity": 10000}
        with self.assertRaisesRegex(ValueError, "missing"):
            allocate(account, {"AAPL": {"market_value": 1}}, {}, Policy())
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            allocate(account, {}, {"AAPL": result("Sell", "Buy")}, Policy())

    def test_daily_loss_and_turnover_limits_block(self):
        with self.assertRaisesRegex(ValueError, "Daily loss"):
            allocate({"equity": 9400, "cash": 9400, "last_equity": 10000}, {},
                     {"AAPL": result("Buy", "Buy")}, Policy())
        with self.assertRaisesRegex(ValueError, "turnover"):
            allocate({"equity": 10000, "cash": 10000, "last_equity": 10000}, {},
                     {"AAPL": result("Buy", "Buy")}, Policy(max_turnover_fraction=.01))


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(os.path.join(self.tmp.name, "portfolio.db"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_cycle_is_durable_and_only_one_can_be_active(self):
        cycle_id = self.store.enqueue(trigger_key="manual-1")
        self.assertEqual(self.store.cycle(cycle_id)["payload"]["progress"]["step"], "queued")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.store.enqueue(trigger_key="manual-2")
        self.assertEqual(Store(self.store.path).cycle(cycle_id)["status"], "queued")

    def test_order_intent_precedes_response_and_remains_recoverable(self):
        cycle = self.store.enqueue()
        self.store.order_intent(cycle, "pf-test", {"symbol": "AAPL"})
        row = Store(self.store.path).orders(cycle)[0]
        self.assertEqual(row["state"], "submitting")
        self.assertEqual(row["response"], {})

    def test_halt_blocks_new_cycles(self):
        self.store.set("halted", True)
        with self.assertRaisesRegex(ValueError, "stop"):
            self.store.enqueue()

    def test_validation_run_is_persistent_and_single_active(self):
        request = HistoricalValidationRequest(
            symbols=["AAPL"], knowledge_cutoff="2025-08-31",
            analysis_dates=["2025-09-04"], horizons=[1, 5],
        )
        run_id = self.store.enqueue_validation(request)
        self.assertEqual(Store(self.store.path).validation(run_id)["status"], "queued")
        with self.assertRaisesRegex(ValueError, "already running"):
            self.store.enqueue_validation(request)


class HistoricalValidationTests(unittest.TestCase):
    def test_portfolio_outcome_matrix(self):
        cases = [
            ("Buy", .02, "matched"),
            ("Buy", -.02, "did_not_match"),
            ("Buy", .005, "inconclusive"),
            ("Sell", -.02, "matched"),
            ("Sell", .02, "did_not_match"),
            ("Sell", .0003, "inconclusive"),
            ("Hold", .0436, "matched"),
            ("Hold", -.005, "matched"),
            ("Hold", -.0211, "did_not_match"),
        ]
        for action, movement, expected in cases:
            with self.subTest(action=action, movement=movement):
                self.assertEqual(outcome_verdict(action, movement), expected)

    def test_next_open_scoring_and_missing_horizon(self):
        bars = [{"date": f"2025-09-{day:02d}", "open": price}
                for day, price in [(5, 100), (8, 105), (9, 110)]]
        scored = score_decision("Buy", bars, bars, [1, 5])
        self.assertAlmostEqual(scored["outcomes"][0]["asset_return"], .05)
        self.assertTrue(scored["outcomes"][0]["correct"])
        self.assertFalse(scored["outcomes"][1]["available"])

    def test_inconclusive_outcome_is_excluded_from_match_rate(self):
        rows = [
            {"status": "completed", "score": {"outcomes": [
                {"available": True, "verdict": "inconclusive", "correct": None,
                 "decision_return": .005, "alpha": .001},
                {"available": True, "verdict": "matched", "correct": True,
                 "decision_return": .02, "alpha": .01},
            ]}},
        ]

        summary = summarize(rows)

        self.assertEqual(summary["outcomes_scored"], 2)
        self.assertEqual(summary["outcomes_decisive"], 1)
        self.assertEqual(summary["outcomes_inconclusive"], 1)
        self.assertEqual(summary["directional_accuracy"], 1)

    def test_keep_uses_the_retained_positions_return(self):
        bars = [{"date": "2025-09-05", "open": 100},
                {"date": "2025-09-08", "open": 104.36}]

        outcome = score_decision("Hold", bars, bars, [1])["outcomes"][0]

        self.assertEqual(outcome["verdict"], "matched")
        self.assertTrue(outcome["correct"])
        self.assertAlmostEqual(outcome["decision_return"], .0436)

    def test_current_matrix_recalculates_stored_results_without_rerunning_research(self):
        payload = {"results": [
            {"status": "completed", "recommendation": {"action": "Hold"},
             "score": {"outcomes": [{"horizon": 5, "available": True,
                                       "asset_return": .0436, "correct": False,
                                       "decision_return": 0, "alpha": .01}]}},
            {"status": "completed", "recommendation": {"action": "Sell"},
             "score": {"outcomes": [{"horizon": 5, "available": True,
                                       "asset_return": .0003, "correct": False,
                                       "decision_return": -.0003, "alpha": 0}]}},
        ], "summary": {"directional_accuracy": 0}}

        updated = apply_outcome_matrix(payload)

        outcomes = [row["score"]["outcomes"][0] for row in updated["results"]]
        self.assertEqual(outcomes[0]["verdict"], "matched")
        self.assertEqual(outcomes[1]["verdict"], "inconclusive")
        self.assertEqual(updated["summary"]["directional_accuracy"], 1)
        self.assertEqual(updated["summary"]["outcomes_inconclusive"], 1)

    def test_validator_reports_progress_and_continues_failed_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(os.path.join(directory, "validation.db"))
            request = HistoricalValidationRequest(
                symbols=["AAPL", "MSFT"], knowledge_cutoff="2025-08-31",
                analysis_dates=["2025-09-04"], horizons=[1], x_posts_mode="disabled",
            )
            run_id = store.enqueue_validation(request)

            class Research:
                def analyze(self, payload, on_job, stopped=lambda: False):
                    on_job("job-" + payload["symbol"])
                    if payload["symbol"] == "MSFT":
                        raise RemoteError("failed")
                    self.payload = payload
                    return {"recommendation": {"action": "Buy", "rating": "Buy"}}

            class Broker:
                def daily_bars(self, ticker, start_date, sessions):
                    return [{"date": "2025-09-05", "open": 100},
                            {"date": "2025-09-08", "open": 103}]

            research = Research()
            HistoricalValidator(store, Broker(), research).run(run_id)
            row = store.validation(run_id)
            self.assertEqual(row["status"], "completed")
            self.assertEqual(row["payload"]["progress"]["percent"], 100)
            self.assertEqual(row["payload"]["summary"]["decisions_completed"], 1)
            self.assertEqual(row["payload"]["summary"]["decisions_failed"], 1)
            self.assertEqual(research.payload["options"]["x_posts_mode"], "disabled")


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(os.path.join(self.tmp.name, "engine.db"))
        self.now = lambda: datetime(2026, 10, 1, 15, tzinfo=timezone.utc)

    def tearDown(self):
        self.tmp.cleanup()

    def cycle(self, **changes):
        policy = Policy(selected_symbols=["NEW"], use_screener_regime=False, **changes)
        self.store.save_policy(policy)
        return self.store.enqueue(policy=policy)

    def test_sells_fill_and_reconcile_before_buys(self):
        broker = FakeBroker()
        cycle = self.cycle()
        Engine(self.store, broker, FakeResearch(), FakeScreener(), now=self.now,
               sleep=lambda _: None).run(cycle)
        self.assertEqual(self.store.cycle(cycle)["status"], "completed")
        self.assertEqual([r["side"] for r in broker.submitted], ["sell", "buy"])
        self.assertNotIn("OLD", broker.position_rows)
        self.assertIn("NEW", broker.position_rows)
        progress = self.store.cycle(cycle)["payload"]["progress"]
        self.assertEqual(progress["step"], "completed")
        self.assertEqual(progress["percent"], 100)
        self.assertEqual(progress["completed"], progress["total"])
        self.assertEqual(progress["trades_completed"], 2)

    def test_analysis_progress_identifies_current_symbol_and_total(self):
        observed = []
        store = self.store

        class ProgressResearch(FakeResearch):
            def analyze(self, payload, on_job, stopped):
                result_row = super().analyze(payload, on_job, stopped)
                observed.append(store.cycle(cycle)["payload"]["progress"].copy())
                return result_row

        cycle = self.cycle(execution_enabled=False)
        Engine(self.store, FakeBroker(), ProgressResearch(), FakeScreener(),
               now=self.now).run(cycle)
        self.assertTrue(observed)
        self.assertEqual(observed[0]["step"], "analyzing")
        self.assertEqual(observed[0]["symbol"], "NEW")
        self.assertEqual(observed[0]["total"], 2)

    def test_preview_builds_plan_without_orders(self):
        broker = FakeBroker()
        cycle = self.cycle(execution_enabled=False)
        Engine(self.store, broker, FakeResearch(), FakeScreener(), now=self.now).run(cycle)
        self.assertEqual(self.store.cycle(cycle)["status"], "preview")
        self.assertEqual(broker.submitted, [])

    def test_closed_market_and_existing_order_fail_before_research(self):
        for broker, message in ((FakeBroker(open_market=False), "closed"),
                                (FakeBroker(existing_orders=True), "open orders")):
            cycle = self.cycle()
            Engine(self.store, broker, FakeResearch(), FakeScreener(), now=self.now).run(cycle)
            row = self.store.cycle(cycle)
            self.assertEqual(row["status"], "failed")
            self.assertIn(message, row["payload"]["error"])

    def test_partial_fill_stops_cycle_for_review(self):
        broker = FakeBroker(fill_status="canceled")
        cycle = self.cycle()
        Engine(self.store, broker, FakeResearch(), FakeScreener(), now=self.now,
               sleep=lambda _: None).run(cycle)
        self.assertEqual(self.store.cycle(cycle)["status"], "needs_review")
        self.assertEqual(len(broker.submitted), 1)

    def test_account_identity_is_pinned(self):
        broker = FakeBroker()
        engine = Engine(self.store, broker, FakeResearch(), FakeScreener(), now=self.now)
        engine.book()
        broker.account_row["id"] = "different-paper-account"
        with self.assertRaisesRegex(ValueError, "changed"):
            engine.book()

    def test_discovery_merges_sources_deduplicates_and_filters(self):
        broker = FakeBroker(untradable={"BAD"})
        research = FakeResearch([
            {"symbol": "DUP", "score": 95, "data_completeness": .9},
            {"symbol": "ACT", "score": 80, "data_completeness": .8},
            {"symbol": "THIN", "score": 90, "data_completeness": .2},
        ])
        screener = FakeScreener(
            candidates=[{"symbol": "SCR", "composite_score": 92},
                        {"symbol": "DUP", "composite_score": 85},
                        {"symbol": "LOW", "composite_score": 69},
                        {"symbol": "BAD", "composite_score": 99}],
            leaders=[{"symbol": "LDR", "composite_score": 90}],
        )
        cycle = self.cycle(execution_enabled=False, stock_screener_candidate_limit=3,
                           alpaca_activity_candidate_limit=2, max_turnover_fraction=1)
        Engine(self.store, broker, research, screener, now=self.now).run(cycle)
        row = self.store.cycle(cycle)
        self.assertEqual(row["status"], "preview")
        self.assertEqual(set(research.analyzed), {"OLD", "NEW", "SCR", "LDR", "DUP", "ACT"})
        selected = row["payload"]["discovery"]["selected"]
        self.assertEqual([item["symbol"] for item in selected[:3]], ["SCR", "LDR", "DUP"])
        self.assertEqual([item["symbol"] for item in selected[3:]], ["ACT"])
        self.assertEqual(selected[3]["data_completeness"], .8)
        self.assertTrue(any(item.get("symbol") == "BAD" for item in
                            row["payload"]["discovery"]["excluded"]))

    def test_stock_candidates_have_priority_when_symbol_capacity_is_full(self):
        broker = FakeBroker()
        research = FakeResearch([
            {"symbol": "ACT", "score": 99, "data_completeness": 1},
        ])
        screener = FakeScreener(candidates=[{"symbol": "SCR", "composite_score": 80}])
        cycle = self.cycle(execution_enabled=False, max_symbols=3,
                           stock_screener_candidate_limit=1,
                           alpaca_activity_candidate_limit=1)
        Engine(self.store, broker, research, screener, now=self.now).run(cycle)
        self.assertEqual(set(research.analyzed), {"OLD", "NEW", "SCR"})

    def test_stale_or_after_hours_discovery_fails_closed(self):
        cases = (
            (FakeResearch(as_of="2026-10-01T14:00:00+00:00"), FakeScreener(), "activity scan"),
            (FakeResearch(session="outside_regular_hours"), FakeScreener(), "outside regular"),
            (FakeResearch(), FakeScreener(snapshot_date="2026-09-20"), "candidates are stale"),
        )
        for research, screener, message in cases:
            cycle = self.cycle()
            Engine(self.store, FakeBroker(), research, screener, now=self.now).run(cycle)
            row = self.store.cycle(cycle)
            self.assertEqual(row["status"], "failed")
            self.assertIn(message, row["payload"]["error"])


class WorkerTests(unittest.TestCase):
    def test_due_schedule_is_coalesced_and_run_once(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(os.path.join(directory, "worker.db"))
            policy = Policy(selected_symbols=["AAPL"], schedule_enabled=True,
                            schedule_mode="interval", interval_minutes=15,
                            use_screener_regime=False, execution_enabled=False)
            store.save_policy(policy)
            store.set("next_run", "2026-10-01T14:59:00+00:00")

            class ScheduledEngine:
                broker = FakeBroker()
                ran = []

                def reconcile(self, cycle_id):
                    raise AssertionError("Nothing should require reconciliation")

                def run(self, cycle_id):
                    self.ran.append(cycle_id)
                    cycle = store.cycle(cycle_id)
                    store.update(cycle_id, "preview", cycle["payload"])

            engine = ScheduledEngine()
            with patch("portfolio_service.worker.utcnow", return_value=datetime(
                    2026, 10, 1, 15, tzinfo=timezone.utc)):
                tick(store, engine)
                tick(store, engine)
            self.assertEqual(len(engine.ran), 1)
            self.assertEqual(len(store.cycles()), 1)


class ClientTests(unittest.TestCase):
    def test_broker_is_hardwired_to_paper_and_requires_explicit_feed(self):
        env = {"ALPACA_USE_PAPER": "true", "ALPACA_API_KEY": "key",
               "ALPACA_SECRET_KEY": "secret", "ALPACA_DATA_FEED": "iex"}
        with patch.dict(os.environ, env, clear=False):
            broker = PaperBroker(session=FakeSession([]))
            self.assertEqual(broker.base, "https://paper-api.alpaca.markets")
        with patch.dict(os.environ, {**env, "ALPACA_USE_PAPER": "false"}, clear=False):
            with self.assertRaisesRegex(ValueError, "PAPER"):
                PaperBroker(session=FakeSession([]))

    def test_no_write_retry_after_ambiguous_failure(self):
        session = FakeSession([FakeResponse(500, {})])
        env = {"ALPACA_USE_PAPER": "true", "ALPACA_API_KEY": "key",
               "ALPACA_SECRET_KEY": "secret", "ALPACA_DATA_FEED": "iex"}
        with patch.dict(os.environ, env, clear=False):
            with self.assertRaises(RemoteError):
                PaperBroker(session=session).submit({"symbol": "AAPL"})
        self.assertEqual(len(session.calls), 1)

    def test_service_urls_reject_embedded_credentials(self):
        with patch.dict(os.environ, {"TRADINGAGENTS_API_URL": "https://user:pass@example.com"}):
            with self.assertRaises(ValueError):
                ResearchClient()

    def test_activity_scanner_contract_and_bounds(self):
        valid = {"schema_version": "1.0", "market": "US", "session": "regular",
                 "as_of": "2026-10-01T15:00:00+00:00", "candidates": []}
        session = FakeSession([FakeResponse(body=valid), FakeResponse(body=[])])
        with patch.dict(os.environ, {"TRADINGAGENTS_API_URL": "http://research"}):
            client = ResearchClient(session=session)
            self.assertEqual(client.scan(5), valid)
            self.assertEqual(session.calls[0][2]["json"], {"top_n": 5})
            with self.assertRaises(RemoteError):
                client.scan(5)
            with self.assertRaisesRegex(ValueError, "between 1 and 50"):
                client.scan(51)

    def test_screener_resource_allowlist_and_symbol_encoding(self):
        session = FakeSession([FakeResponse(body={"symbol": "BRK.B"})])
        with patch.dict(os.environ, {"STOCK_SCREENER_API_URL": "http://scanner/api/v1"}):
            body = ScreenerClient(session=session).read("symbol", "BRK.B")
        self.assertEqual(body["symbol"], "BRK.B")
        self.assertIn("/stocks/BRK.B/decision-dashboard", session.calls[0][1])

    def test_screener_scan_uses_typed_us_market_contract(self):
        session = FakeSession([FakeResponse(body={"scan_id": "scan-1"})])
        with patch.dict(os.environ, {"STOCK_SCREENER_API_URL": "http://scanner/api/v1"}):
            ScreenerClient(session=session).scan({"market": "US", "screeners": ["minervini"]})
        sent = session.calls[0][2]["json"]
        self.assertEqual(sent["universe_def"], {"type": "market", "market": "US"})
        self.assertNotIn("market", sent)
        with patch.dict(os.environ, {"STOCK_SCREENER_API_URL": "http://scanner/api/v1"}):
            with self.assertRaisesRegex(ValueError, "US market"):
                ScreenerClient(session=FakeSession([])).scan(
                    {"universe_def": {"type": "market", "market": "HK"}}
                )


class WebTests(unittest.TestCase):
    def setUp(self):
        from flask import Flask
        self.tmp = tempfile.TemporaryDirectory()
        self.app = Flask(__name__, template_folder="../webui/templates")
        self.store = Store(os.path.join(self.tmp.name, "web.db"))
        self.broker = FakeBroker()
        with patch.dict(os.environ, {"DASHBOARD_USERNAME": "admin",
                                     "DASHBOARD_PASSWORD": "this-is-a-long-test-password"}):
            register(self.app, self.store, broker_factory=lambda: self.broker)
        self.client = self.app.test_client()
        import base64
        self.auth = {"Authorization": "Basic " + base64.b64encode(
            b"admin:this-is-a-long-test-password").decode()}

    def tearDown(self):
        self.tmp.cleanup()

    def test_authentication_and_csrf_header(self):
        self.assertEqual(self.client.get("/api/portfolio/status").status_code, 401)
        self.assertEqual(self.client.get("/api/portfolio/status", headers=self.auth).status_code, 200)
        self.assertEqual(self.client.post("/api/portfolio/halt", headers=self.auth,
                                          json={"halted": True}).status_code, 403)
        headers = {**self.auth, "X-Portfolio-Request": "1"}
        self.assertEqual(self.client.post("/api/portfolio/halt", headers=headers,
                                          json={"halted": True}).status_code, 200)

    def test_overview_returns_plain_account_and_holding_summary(self):
        response = self.client.get("/api/portfolio/overview", headers=self.auth)
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertTrue(body["paper"])
        self.assertEqual(body["account"]["portfolio_value"], 10000)
        self.assertEqual(body["account"]["invested"], 1000)
        self.assertEqual(body["positions"][0]["symbol"], "OLD")
        self.assertTrue(body["market_open"])
        self.assertIn("next_market_open", body)
        self.assertNotIn("id", body["account"])

    def test_portfolio_page_uses_plain_language_navigation(self):
        response = self.client.get("/portfolio", headers=self.auth)
        self.assertEqual(response.status_code, 200)
        page = response.get_data(as_text=True)
        for label in ("My Portfolio", "Home", "Opportunities", "Activity", "Historical validation", "Settings",
                      "Check market now", "progress-panel", "companies analyzed"):
            self.assertIn(label, page)
        self.assertNotIn("Run a custom stock-screener scan", page)

    def test_integrated_root_redirects_to_simple_portfolio(self):
        from flask import Flask
        app = Flask(__name__, template_folder="../webui/templates")
        with patch.dict(os.environ, {"PORTFOLIO_ENABLED": "true",
                                     "DASHBOARD_PASSWORD": "this-is-a-long-test-password"}):
            register(app, self.store, broker_factory=lambda: self.broker)
        response = app.test_client().get("/", headers=self.auth)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/portfolio")

    def test_http_enqueue_to_mocked_complete_cycle_and_status(self):
        headers = {**self.auth, "X-Portfolio-Request": "1"}
        policy = Policy(selected_symbols=["NEW"], use_screener_regime=False).model_dump()
        self.assertEqual(self.client.post("/api/portfolio/policy", headers=headers,
                                          json=policy).status_code, 200)
        queued = self.client.post("/api/portfolio/run", headers=headers, json={})
        self.assertEqual(queued.status_code, 202)
        cycle_id = queued.get_json()["cycle_id"]
        Engine(self.store, FakeBroker(), FakeResearch(), FakeScreener(),
               now=lambda: datetime(2026, 10, 1, 15, tzinfo=timezone.utc),
               sleep=lambda _: None).run(cycle_id)
        status = self.client.get("/api/portfolio/status", headers=self.auth).get_json()
        self.assertEqual(status["cycles"][0]["status"], "completed")
        self.assertEqual([row["request"]["side"] for row in status["orders"]], ["sell", "buy"])

    def test_non_us_scan_is_rejected_before_remote_call(self):
        headers = {**self.auth, "X-Portfolio-Request": "1"}
        response = self.client.post("/api/portfolio/scans", headers=headers,
                                    json={"market": "HK"})
        self.assertEqual(response.status_code, 400)

    def test_historical_validation_endpoint_queues_validated_grid(self):
        headers = {**self.auth, "X-Portfolio-Request": "1"}
        response = self.client.post("/api/portfolio/validations", headers=headers, json={
            "symbols": ["AAPL", "MSFT"], "knowledge_cutoff": "2025-08-31",
            "analysis_dates": ["2025-09-04"], "horizons": [1, 5],
            "x_posts_mode": "disabled",
        })
        self.assertEqual(response.status_code, 202)
        latest = self.client.get("/api/portfolio/validations/latest", headers=self.auth).get_json()
        self.assertEqual(latest["validation"]["payload"]["progress"]["total"], 2)

    def test_historical_validation_rejects_future_or_oversized_grid(self):
        headers = {**self.auth, "X-Portfolio-Request": "1"}
        response = self.client.post("/api/portfolio/validations", headers=headers, json={
            "symbols": ["AAPL"], "knowledge_cutoff": "2025-08-31",
            "analysis_dates": ["2099-09-04"], "horizons": [5],
            "x_posts_mode": "disabled",
        })
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
