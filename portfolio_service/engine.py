"""Research the complete book, allocate, sell, reconcile, and then buy."""
from __future__ import annotations

import math
import time
from datetime import date
from zoneinfo import ZoneInfo

from .models import Policy, allocate, number, symbol, timestamp, utcnow

TERMINAL = {"filled", "canceled", "expired", "rejected", "replaced"}


class Engine:
    def __init__(self, store, broker, research, screener, *, now=utcnow, sleep=time.sleep):
        self.store, self.broker, self.research, self.screener = store, broker, research, screener
        self.now, self.sleep = now, sleep

    def stopped(self):
        return self.store.get("halted", False)

    def check(self):
        if self.stopped():
            raise ValueError("Emergency stop requested")
        if not self.broker.clock().get("is_open"):
            raise ValueError("US market is closed")

    def progress(self, cycle_id, data, status, step, message, percent, **details):
        data["progress"] = {"step": step, "message": message,
                            "percent": min(100, max(0, round(percent))), **details}
        self.store.update(cycle_id, status, data)

    def book(self):
        account = self.broker.account()
        pinned = self.store.get("account_id")
        if pinned and pinned != account["id"]:
            raise ValueError("Paper account changed; use a separate portfolio database")
        self.store.set("account_id", account["id"])
        positions = {}
        for position in self.broker.positions():
            ticker = symbol(position["symbol"])
            if position.get("asset_class") != "us_equity" or number(position["qty"], minimum=-1e99) < 0:
                raise ValueError("Unsupported asset or short holding requires review")
            for key in ("qty", "market_value", "avg_entry_price"):
                number(position[key])
            if ticker in positions:
                raise ValueError("Duplicate broker position")
            positions[ticker] = position
        return account, positions

    def learning(self, account):
        history = self.store.evaluations(account["id"])
        if len(history) < 10:
            return 1.0, {"samples": len(history), "mode": "collecting", "risk_multiplier": 1.0}
        # Outcomes can reduce exposure. No automatic increase beyond user limits,
        # prompt mutation, or performance claim. Deposits/withdrawals must be
        # reviewed; this conservative control only uses account drawdown.
        peak = max(number(r["equity"], minimum=.01) for r in history)
        drawdown = max(0, 1 - number(account["equity"]) / peak)
        multiplier = max(.5, 1 - 5 * drawdown)
        return multiplier, {"samples": len(history), "mode": "drawdown_control",
                            "drawdown": drawdown, "risk_multiplier": multiplier}

    def freshness(self, research, policy):
        for ticker, result in research.items():
            age = (self.now() - timestamp(result["completed_at"])).total_seconds()
            if age < -5 or age > policy.max_research_age_seconds:
                raise ValueError(f"Stale research for {ticker}; rerun the complete cycle")

    def discovery(self, policy, snapshot, mandatory):
        """Merge bounded discovery sources; stock-screener receives first priority."""
        selected = list(sorted(mandatory))
        evidence = {"mandatory": selected.copy(), "selected": [], "excluded": []}
        today = self.now().astimezone(ZoneInfo("America/New_York")).date()

        def add(rows, source, score_field, minimum, limit, *, completeness=False):
            accepted = 0
            ranked = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                try:
                    ticker = symbol(row.get("symbol"))
                    score = number(row.get(score_field))
                except (TypeError, ValueError):
                    evidence["excluded"].append({"source": source, "reason": "invalid_candidate"})
                    continue
                if score > 100 or score < minimum:
                    continue
                row_completeness = None
                if completeness:
                    try:
                        row_completeness = number(row.get("data_completeness", 0))
                    except (TypeError, ValueError):
                        row_completeness = 0
                    if row_completeness < .5:
                        evidence["excluded"].append({"symbol": ticker, "source": source,
                                                     "reason": "insufficient_data"})
                        continue
                ranked.append((score, ticker, row, row_completeness))
            for score, ticker, row, row_completeness in sorted(
                    ranked, key=lambda item: (-item[0], item[1])):
                if accepted >= limit or len(selected) >= policy.max_symbols:
                    break
                if ticker in selected:
                    continue
                row_source = row.get("discovery_source", source)
                try:
                    self.broker.asset(ticker)
                except ValueError:
                    evidence["excluded"].append({"symbol": ticker, "source": row_source,
                                                 "reason": "not_tradable_us_equity"})
                    continue
                selected.append(ticker)
                accepted += 1
                record = {"symbol": ticker, "source": row_source, "score": score}
                if completeness:
                    record["data_completeness"] = row_completeness
                    record["warnings"] = row.get("warnings") or []
                elif row.get("rating") is not None:
                    record["rating"] = row["rating"]
                evidence["selected"].append(record)

        if policy.include_stock_screener_candidates:
            freshness = snapshot.get("freshness") or {}
            snapshot_date = (freshness.get("snapshot_as_of_date")
                             or (snapshot.get("market_health_exposure") or {}).get("date"))
            try:
                age = (today - date.fromisoformat(snapshot_date)).days
            except (TypeError, ValueError):
                raise ValueError("Stock-screener candidate snapshot has no valid date") from None
            if not 0 <= age <= 4:
                raise ValueError("Stock-screener candidates are stale")
            candidate_rows = []
            for section, source in (("top_candidates", "stock_screener_candidate"),
                                    ("leaders", "stock_screener_leader")):
                rows = (snapshot.get(section) or {}).get("rows") or []
                for row in rows:
                    if isinstance(row, dict):
                        candidate_rows.append({**row, "_source": source})
            # Preserve the originating panel; add() validates and ranks the combined list.
            for row in candidate_rows:
                row.setdefault("discovery_source", row.pop("_source", "stock_screener_candidate"))
            add(candidate_rows, "stock_screener_candidate", "composite_score",
                policy.stock_screener_min_score, policy.stock_screener_candidate_limit)

        if policy.include_alpaca_activity_candidates and len(selected) < policy.max_symbols:
            scan = self.research.scan(top_n=min(50, policy.alpaca_activity_candidate_limit * 2))
            age = (self.now() - timestamp(scan.get("as_of"))).total_seconds()
            if age < -5 or age > policy.max_activity_scan_age_seconds:
                raise ValueError("Alpaca activity scan is stale")
            if scan.get("session") != "regular":
                raise ValueError("Alpaca activity scan is outside regular market hours")
            add(scan["candidates"], "alpaca_activity", "score",
                policy.alpaca_activity_min_score, policy.alpaca_activity_candidate_limit,
                completeness=True)
            evidence["alpaca_metadata"] = scan.get("metadata", {})
            evidence["alpaca_feed"] = scan.get("feed")
            evidence["alpaca_as_of"] = scan.get("as_of")
        return selected, evidence

    def run(self, cycle_id):
        cycle = self.store.cycle(cycle_id)
        if not cycle or cycle["status"] != "queued":
            raise ValueError("Only queued cycles may start")
        data = cycle["payload"]
        policy = Policy.model_validate(data["policy"])
        try:
            self.check()
            self.progress(cycle_id, data, "researching", "account",
                          "Checking your paper account and current investments", 10,
                          completed=0, total=0)
            account, positions = self.book()
            if self.broker.open_orders():
                raise ValueError("Existing open orders require reconciliation before this cycle")
            mandatory = set(policy.selected_symbols) | set(positions)
            if len(mandatory) > policy.max_symbols:
                raise ValueError("Selected and held symbols exceed the configured maximum")
            data.update(account=account, positions=positions, research={}, jobs={})
            self.progress(cycle_id, data, "researching", "discovering",
                          "Finding strong market opportunities", 20,
                          completed=0, total=0)
            exposure = 1.0
            snapshot = None
            if policy.use_screener_regime or policy.include_stock_screener_candidates:
                snapshot = self.screener.daily()
                data["screener_context"] = snapshot
            if policy.use_screener_regime:
                regime = snapshot.get("market_health_exposure")
                today = self.now().astimezone(ZoneInfo("America/New_York")).date()
                if not regime or not 0 <= (today - date.fromisoformat(regime["date"])).days <= 4:
                    raise ValueError("Missing or stale stock-screener exposure context")
                exposure = min(1.0, number(regime["exposure_score"]) / 100)
            tickers, discovery = self.discovery(policy, snapshot or {}, mandatory)
            if not tickers:
                raise ValueError("No mandatory or eligible discovery symbols")
            data["discovery"] = discovery
            total = len(tickers)
            self.progress(cycle_id, data, "researching", "analyzing",
                          f"Preparing to analyze {total} companies", 25,
                          completed=0, total=total)
            context = {"cash": number(account["cash"]), "currency": "USD", "positions": [
                {"ticker": s, "quantity": number(p["qty"]), "average_price": number(p["avg_entry_price"])}
                for s, p in positions.items()]}
            trade_date = self.now().astimezone(ZoneInfo("America/New_York")).date().isoformat()
            for ticker in sorted(tickers):
                self.check()
                self.broker.asset(ticker)
                completed = len(data["research"])
                self.progress(cycle_id, data, "researching", "analyzing",
                              f"Analyzing {ticker} with TradingAgents",
                              25 + 50 * completed / max(total, 1),
                              completed=completed, total=total, symbol=ticker)
                def on_job(job_id, ticker=ticker):
                    data["jobs"][ticker] = job_id
                    self.progress(cycle_id, data, "researching", "analyzing",
                                  f"Analyzing {ticker} with TradingAgents",
                                  25 + 50 * len(data["research"]) / max(total, 1),
                                  completed=len(data["research"]), total=total, symbol=ticker)
                payload = {"symbol": ticker, "trade_date": trade_date, "asset_type": "stock",
                           "portfolio": context, "options": {"output_language": "English",
                           "max_debate_rounds": 2, "max_risk_rounds": 2, "checkpoint_enabled": True}}
                data["research"][ticker] = self.research.analyze(payload, on_job, self.stopped)
                completed = len(data["research"])
                self.progress(cycle_id, data, "researching", "analyzing",
                              f"Finished analyzing {ticker}",
                              25 + 50 * completed / max(total, 1),
                              completed=completed, total=total, symbol=ticker)
            self.freshness(data["research"], policy)
            self.progress(cycle_id, data, "researching", "planning",
                          "Building a balanced portfolio plan", 80,
                          completed=total, total=total)
            fresh_account, fresh_positions = self.book()
            if self.quantities(positions) != self.quantities(fresh_positions):
                raise ValueError("Holdings changed during research")
            multiplier, learning = self.learning(fresh_account)
            data["learning"] = learning
            data["plan"] = allocate(fresh_account, fresh_positions, data["research"], policy,
                                     exposure=exposure, learning=policy.investment_fraction * multiplier
                                     if policy.learning_enabled else 1.0)
            if not policy.execution_enabled:
                self.progress(cycle_id, data, "preview", "completed",
                              "Recommendations are ready for review", 100,
                              completed=total, total=total)
                return
            self.progress(cycle_id, data, "executing", "trading",
                          "Preparing safe paper trades", 88,
                          completed=total, total=total, trades_completed=0)
            self.execute(cycle_id, data, policy, fresh_positions)
            after, after_positions = self.book()
            data["after"] = {"account": after, "positions": after_positions}
            self.store.evaluate(after, {"cycle_id": cycle_id, "learning": learning,
                                       "policy": policy.model_dump(), "plan": data["plan"]})
            self.progress(cycle_id, data, "completed", "completed",
                          "Portfolio check and paper trades completed", 100,
                          completed=total, total=total,
                          trades_completed=data.get("progress", {}).get("trades_completed", 0))
        except Exception as exc:
            data["error"] = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
            status = "needs_review" if self.store.orders(cycle_id) else "failed"
            previous = data.get("progress", {})
            self.progress(cycle_id, data, status, "stopped",
                          "A safety check stopped this market check",
                          previous.get("percent", 0),
                          completed=previous.get("completed", 0),
                          total=previous.get("total", 0))

    @staticmethod
    def quantities(positions):
        return {s: number(p["qty"]) for s, p in positions.items() if number(p["qty"]) > 0}

    @staticmethod
    def same_quantities(left, right):
        symbols = set(left) | set(right)
        return all(abs(left.get(s, 0) - right.get(s, 0)) <= 0.000001 for s in symbols)

    def wait_for_positions(self, expected):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            _, positions = self.book()
            if self.same_quantities(self.quantities(positions), expected):
                return
            self.sleep(.5)
        raise ValueError("Filled order is not reflected in the broker positions; reconciliation required")

    def order(self, cycle_id, request):
        client_id = request["client_order_id"]
        self.store.order_intent(cycle_id, client_id, request)
        # Persist intent BEFORE transmitting. A timeout is ambiguous and is
        # reconciled by client ID; it never triggers another POST.
        response = self.broker.submit(request)
        self.store.order_result(client_id, response)
        deadline = time.monotonic() + 90
        while response.get("status") not in TERMINAL:
            if self.stopped() or time.monotonic() >= deadline:
                self.broker.cancel(response["id"])
                raise ValueError("Order cancellation requested; reconciliation required")
            self.sleep(1)
            response = self.broker.lookup(client_id)
            if not response:
                raise ValueError("Broker order status is uncertain")
            self.store.order_result(client_id, response)
        if response["status"] != "filled":
            raise ValueError("Order did not fill completely; remaining allocation paused")
        return response

    def execute(self, cycle_id, data, policy, initial_positions):
        targets = data["plan"]["targets"]
        expected = self.quantities(initial_positions)
        for side in ("sell", "buy"):
            for ticker in sorted(targets):
                self.check()
                self.freshness(data["research"], policy)
                account, positions = self.book()
                if not self.same_quantities(self.quantities(positions), expected) or self.broker.open_orders():
                    raise ValueError("Concurrent broker activity detected")
                # Reapply the daily-loss gate immediately before each order.
                if 1 - number(account["equity"]) / number(account["last_equity"], minimum=.01) >= policy.max_daily_loss_fraction:
                    raise ValueError("Daily loss limit reached during execution")
                asset = self.broker.asset(ticker)
                q = self.broker.quote(ticker, policy, self.now())
                position = positions.get(ticker, {})
                held_qty = number(position.get("qty", 0))
                held_value = held_qty * q["bid"]
                delta = targets[ticker] - held_value
                if (side == "sell" and delta >= -policy.min_trade_dollars) or (side == "buy" and delta <= policy.min_trade_dollars):
                    continue
                request = {"symbol": ticker, "side": side, "type": "market", "time_in_force": "day",
                           "client_order_id": f"pf-{cycle_id}-{side[0]}-{ticker}"}
                if side == "sell":
                    qty = min(held_qty, -delta / q["bid"])
                    if targets[ticker] == 0:
                        qty = held_qty
                    qty = math.floor(qty * (1e6 if asset.get("fractionable") else 1)) / (1e6 if asset.get("fractionable") else 1)
                    if qty <= 0:
                        continue
                    request["qty"] = format(qty, ".6f")
                else:
                    reserve = max(data["plan"]["minimum_cash"], number(account["equity"]) * policy.cash_reserve_fraction)
                    gross = sum(number(p["market_value"]) for p in positions.values())
                    budget = min(delta, number(account["cash"]) - reserve, number(account["buying_power"]),
                                 number(account["equity"]) * data["plan"]["deployment_fraction"] - gross,
                                 number(account["equity"]) * policy.max_position_fraction - held_value)
                    budget = math.floor(max(0, budget) / (1 + policy.slippage_buffer_fraction) * 100) / 100
                    if budget < policy.min_trade_dollars:
                        continue
                    if asset.get("fractionable"):
                        request["notional"] = f"{budget:.2f}"
                    else:
                        qty = math.floor(budget / (q["ask"] * (1 + policy.slippage_buffer_fraction)))
                        if qty < 1:
                            continue
                        # Limit orders bound spend for whole-share purchases.
                        request.update(type="limit", qty=str(qty), limit_price=f"{q['ask']:.2f}")
                trade_count = data.get("progress", {}).get("trades_completed", 0)
                self.progress(cycle_id, data, "executing", "trading",
                              f"Submitting paper {side} for {ticker}",
                              min(96, 88 + trade_count * 2),
                              completed=len(data.get("research", {})),
                              total=len(data.get("research", {})), symbol=ticker,
                              trade_side=side, trades_completed=trade_count)
                filled = self.order(cycle_id, request)
                trade_count = data.get("progress", {}).get("trades_completed", 0) + 1
                self.progress(cycle_id, data, "executing", "trading",
                              f"Completed paper {side} for {ticker}",
                              min(98, data.get("progress", {}).get("percent", 88) + 2),
                              completed=len(data.get("research", {})),
                              total=len(data.get("research", {})), symbol=ticker,
                              trade_side=side, trades_completed=trade_count)
                qty = number(filled["filled_qty"])
                expected[ticker] = round(expected.get(ticker, 0) + (qty if side == "buy" else -qty), 6)
                if expected[ticker] <= .0000001:
                    expected.pop(ticker)
                self.wait_for_positions(expected)

    def reconcile(self, cycle_id):
        cycle = self.store.cycle(cycle_id)
        if not cycle or cycle["status"] not in {"needs_review", "executing", "reconciling", "researching"}:
            return
        self.book()  # verify pinned account before order lookups
        unresolved = []
        for order in self.store.orders(cycle_id):
            response = self.broker.lookup(order["client_id"])
            if response:
                self.store.order_result(order["client_id"], response)
                if self.stopped() and response.get("status") not in TERMINAL:
                    self.broker.cancel(response["id"])
                if response.get("status") not in TERMINAL:
                    unresolved.append(order["client_id"])
            else:
                # A definitive 404 is recorded, never retried as a new order.
                self.store.order_result(order["client_id"], {"status": "not_found"})
        data = cycle["payload"]
        data["unresolved_orders"] = unresolved
        data["recovery"] = "No orders resubmitted; start a new cycle after reconciliation"
        self.store.update(cycle_id, "needs_review" if unresolved else "interrupted", data)
