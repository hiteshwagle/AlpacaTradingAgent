"""
webui/components/analysis.py
"""

from datetime import datetime

from integrations.tradingagents_api import (
    TradingAgentsAPIClient,
    TradingAgentsAPIError,
    build_trade_intent,
    map_analysts,
    reconcile_action,
    result_to_webui_state,
)
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.run_logger import get_run_audit_logger
from tradingagents.dataflows.alpaca_utils import AlpacaUtils, get_alpaca_trading_client
from tradingagents.agents.schemas import trade_intent_action
from webui.utils.state import app_state
from webui.utils.charts import create_chart


def _portfolio_context_from_alpaca():
    """Return a point-in-time broker portfolio, or None if it cannot be verified."""
    try:
        client = get_alpaca_trading_client()
        account = client.get_account()
        positions = client.get_all_positions()
        return {
            "cash": float(account.cash),
            "currency": "USD",
            "positions": [
                {
                    "ticker": position.symbol,
                    "quantity": float(position.qty),
                    "average_price": float(position.avg_entry_price),
                }
                for position in positions
            ],
        }
    except Exception as exc:
        print(f"[ANALYSIS] Alpaca portfolio context unavailable: {exc}")
        return None


def _api_payload(ticker, selected_analysts, depth_rounds, output_language, checkpoint_enabled):
    symbol = ticker.strip().upper()
    return {
        "symbol": symbol,
        "trade_date": datetime.now().strftime("%Y-%m-%d"),
        "asset_type": "crypto" if "/" in symbol else "stock",
        "analysts": map_analysts(selected_analysts),
        "portfolio": _portfolio_context_from_alpaca(),
        "options": {
            "max_debate_rounds": depth_rounds,
            "max_risk_rounds": depth_rounds,
            "output_language": output_language or "English",
            "checkpoint_enabled": bool(checkpoint_enabled),
            "save_reports": True,
        },
    }


def _apply_api_result(current_state, result, selected_analysts):
    final_state = result_to_webui_state(result, selected_analysts)
    reports = current_state["current_reports"]
    reports.update(
        {
            "market_report": final_state["market_report"],
            "sentiment_report": final_state["sentiment_report"],
            "news_report": final_state["news_report"],
            "fundamentals_report": final_state["fundamentals_report"],
            "macro_report": final_state["macro_report"],
            "bull_report": final_state["investment_debate_state"]["bull_history"],
            "bear_report": final_state["investment_debate_state"]["bear_history"],
            "research_manager_report": final_state["investment_plan"],
            "investment_plan": final_state["investment_plan"],
            "trader_investment_plan": final_state["trader_investment_plan"],
            "risky_report": final_state["risk_debate_state"]["risky_history"],
            "safe_report": final_state["risk_debate_state"]["safe_history"],
            "neutral_report": final_state["risk_debate_state"]["neutral_history"],
            "portfolio_decision": final_state["risk_debate_state"]["judge_decision"],
            "final_trade_decision": final_state["final_trade_decision"],
        }
    )
    current_state["investment_debate_state"] = final_state["investment_debate_state"]
    current_state["risk_debate_state"] = final_state["risk_debate_state"]
    current_state["source_urls"] = list(result.get("source_urls") or [])
    return final_state


def execute_trade_after_analysis(ticker, allow_shorts, trade_amount):
    """Execute trade based on analysis results"""
    try:
        print(f"[TRADE] Starting trade execution for {ticker}")

        # Get the current state for this symbol
        state = app_state.get_state(ticker)
        if not state:
            print(f"[TRADE] No state found for {ticker}, skipping trade execution")
            return

        if not state.get("analysis_complete"):
            print(f"[TRADE] Analysis not complete for {ticker}, skipping trade execution")
            print(f"[TRADE] Analysis status: {state.get('analysis_complete', 'Unknown')}")
            return

        print(f"[TRADE] Analysis complete for {ticker}, checking for recommended action")

        analysis_results = state.get("analysis_results") or {}
        api_result = analysis_results.get("api_result")
        trade_intent = None
        recommended_action = reconcile_action(api_result) if api_result else state.get("recommended_action")
        print(f"[TRADE] Direct recommended_action: {recommended_action}")

        if not recommended_action:
            print(
                f"[TRADE] TradingAgents rating/action is absent or inconsistent for {ticker}; "
                "skipping trade execution"
            )
            return

        print(f"[TRADE] Executing trade for {ticker}: {recommended_action} with ${trade_amount}")

        # Portfolio-level sizing: deterministic layer above the per-symbol
        # decision (correlation penalty, inverse-vol sizing, gross exposure
        # cap). Failure-isolated: any problem keeps the requested amount.
        try:
            from tradingagents.dataflows.config import get_config
            from tradingagents.portfolio import (
                PortfolioLimitsConfig,
                adjust_new_position_notional,
                gather_portfolio_state_via_alpaca,
            )

            trade_amount = adjust_new_position_notional(
                symbol=ticker,
                action=recommended_action,
                requested_notional=trade_amount,
                gather_state=lambda: gather_portfolio_state_via_alpaca(ticker),
                config=PortfolioLimitsConfig.from_config(get_config() or {}),
            )
            if trade_amount <= 0:
                print(
                    f"[TRADE] Portfolio layer zeroed the {ticker} order "
                    "(no gross-exposure headroom); skipping execution."
                )
                return
        except Exception as exc:
            print(f"[TRADE] Portfolio sizing unavailable for {ticker}: {exc}")

        # Regime-aware sizing: hostile regimes shrink NEW exposure, never
        # flip the decision. Failure-isolated - any problem keeps the
        # requested amount untouched.
        if str(recommended_action).upper() in ("BUY", "LONG"):
            try:
                from tradingagents.dataflows.config import get_config
                from tradingagents.regime import RegimeConfig, regime_risk_multiplier

                multiplier = regime_risk_multiplier(
                    ticker, config=RegimeConfig.from_config(get_config() or {})
                )
                if multiplier < 1.0:
                    trade_amount = trade_amount * multiplier
                    print(
                        f"[TRADE] Regime filter scaled {ticker} amount to "
                        f"${trade_amount:,.0f} (x{multiplier:.2f})"
                    )
            except Exception as exc:
                print(f"[TRADE] Regime sizing unavailable for {ticker}: {exc}")

        # Get current position. strict=True: a broker outage here must abort
        # the trade instead of reading as NEUTRAL — acting on a guessed
        # NEUTRAL re-buys an existing holding on BUY (pyramiding every loop
        # iteration the outage persists) and skips the exit on SELL.
        try:
            current_position = AlpacaUtils.get_current_position_state(ticker, strict=True)
        except Exception as e:
            print(
                f"[TRADE] Could not verify current position for {ticker} ({e}); "
                "skipping trade execution rather than guessing NEUTRAL"
            )
            state["trading_results"] = {
                "error": (
                    f"Position check failed for {ticker}; trade skipped to avoid "
                    f"acting on an unverified position: {e}"
                )
            }
            return
        print(f"[TRADE] Current position for {ticker}: {current_position}")

        if api_result:
            try:
                trade_intent = build_trade_intent(
                    api_result,
                    symbol=ticker,
                    current_position=current_position,
                    allow_shorts=allow_shorts,
                    trade_date=str(analysis_results.get("date") or ""),
                )
            except TradingAgentsAPIError as exc:
                print(f"[TRADE] {exc}")
                state["trading_results"] = {"error": str(exc)}
                return
            state["final_trade_intent"] = trade_intent
            analysis_results["trade_intent"] = trade_intent
            intent_action = trade_intent_action(trade_intent)
            if intent_action != recommended_action:
                state["trading_results"] = {
                    "error": "Trade intent action did not match the accepted TradingAgents action; trade blocked."
                }
                return

        # API-backed runs always execute a freshly built typed intent. The legacy
        # branch remains only for stored pre-integration runs.
        if trade_intent:
            risk_params = (
                dict(DEFAULT_CONFIG.get("risk_sizing_params") or {})
                if DEFAULT_CONFIG.get("risk_sizing_enabled")
                else None
            )
            result = AlpacaUtils.execute_trade_intent(
                symbol=ticker,
                current_position=current_position,
                trade_intent=trade_intent,
                dollar_amount=trade_amount,
                allow_shorts=allow_shorts,
                risk_params=risk_params,
            )
        else:
            result = AlpacaUtils.execute_trading_action(
                symbol=ticker,
                current_position=current_position,
                signal=recommended_action,
                dollar_amount=trade_amount,
                allow_shorts=allow_shorts
            )

        # Check individual action results and provide detailed feedback
        successful_actions = []
        failed_actions = []

        for action_result in result.get("actions", []):
            if "result" in action_result:
                action_info = action_result["result"]
                if action_info.get("success"):
                    successful_actions.append(f"{action_result['action']}: {action_info.get('message', 'Success')}")
                else:
                    failed_actions.append(f"{action_result['action']} failed: {action_info.get('error', 'Unknown error')}")
            else:
                successful_actions.append(f"{action_result['action']}: {action_result.get('message', 'Action completed')}")

        for warning in result.get("intent_warnings", []):
            print(f"[TRADE] Intent warning: {warning}")

        # Print results based on overall success
        if result.get("success"):
            print(f"[TRADE] Successfully executed trading actions for {ticker}")
            for success in successful_actions:
                print(f"[TRADE] {success}")

            # Store trading results in state for UI display
            state["trading_results"] = result

            # Signal that a trade occurred to trigger Alpaca data refresh
            app_state.signal_trade_occurred()
        else:
            print(f"[TRADE] Trading execution failed for {ticker}")
            for success in successful_actions:
                print(f"[TRADE] {success}")
            for failure in failed_actions:
                print(f"[TRADE] {failure}")
            if result.get("error"):
                print(f"[TRADE] {result['error']}")

            # Store error information
            state["trading_results"] = {
                "error": result.get("error", "One or more trading actions failed"),
                "details": failed_actions,
                "raw_result": result,
            }

    except Exception as e:
        print(f"[TRADE] Error executing trade for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        state = app_state.get_state(ticker)
        if state:
            state["trading_results"] = {"error": f"Trading execution error: {str(e)}"}


def run_analysis(
    ticker,
    selected_analysts,
    research_depth_config,
    allow_shorts,
    quick_llm,
    deep_llm,
    quick_llm_params=None,
    deep_llm_params=None,
    llm_provider="openai",
    backend_url=None,
    output_language="English",
    checkpoint_enabled=False,
    provider_settings=None,
    progress=None,
):
    """Request research from TradingAgents and prepare it for Alpaca controls."""
    run_logger = get_run_audit_logger()
    run_started = False
    final_state = None
    current_state = app_state.get_state(ticker)
    current_date = datetime.now().strftime("%Y-%m-%d")

    try:
        if not current_state:
            raise TradingAgentsAPIError(f"No UI state exists for {ticker}.")
        print(f"Requesting TradingAgents analysis for {ticker} on {current_date}")
        current_state["analysis_running"] = True
        current_state["analysis_complete"] = False
        current_state["analysis_error"] = None

        if isinstance(research_depth_config, dict):
            depth_rounds = research_depth_config.get("rounds", 3)
            depth_level = research_depth_config.get("level", "Medium")
        else:
            depth_rounds = research_depth_config
            depth_map = {1: "Shallow", 3: "Medium", 5: "Deep"}
            depth_level = depth_map.get(research_depth_config, "Medium")

        payload = _api_payload(
            ticker,
            selected_analysts,
            depth_rounds,
            output_language,
            checkpoint_enabled,
        )
        client = TradingAgentsAPIClient()
        audit_config = {
            "research_provider": "tradingagents_api",
            "tradingagents_api_url": client.base_url,
            "selected_analysts": payload["analysts"],
            "research_depth": depth_level,
            "max_debate_rounds": depth_rounds,
            "max_risk_discuss_rounds": depth_rounds,
            "output_language": output_language or "English",
            "allow_shorts": allow_shorts,
        }
        run_logger.start_run(
            symbol=ticker,
            trade_date=current_date,
            config=audit_config,
            metadata={"source": "tradingagents_api"},
        )
        run_started = True

        active_agents = {
            "market": "Market Analyst",
            "social": "Social Analyst",
            "news": "News Analyst",
            "fundamentals": "Fundamentals Analyst",
        }
        if "macro" in selected_analysts:
            active_agents["macro"] = "Macro Analyst"
        for key, label in active_agents.items():
            if key in selected_analysts or (key == "news" and "news" in payload["analysts"]):
                app_state.update_agent_status(label, "in_progress")

        def on_status(job):
            status = str(job.get("status", ""))
            print(f"[ANALYSIS] TradingAgents job {job.get('analysis_id')}: {status}")
            if progress is not None:
                progress(0.05 if status == "queued" else 0.25)
            app_state.needs_ui_update = True

        result = client.analyze(payload, on_status=on_status)
        final_state = _apply_api_result(current_state, result, selected_analysts)
        decision = reconcile_action(result)
        if decision is None:
            print("[ANALYSIS] Remote result is advisory only: rating and action did not reconcile.")

        current_state["recommended_action"] = decision
        current_state["final_trade_intent"] = None
        current_state["analysis_results"] = {
            "ticker": ticker,
            "date": current_date,
            "decision": decision,
            "trade_intent": None,
            "api_result": result,
            "analysis_id": result["_api_metadata"]["analysis_id"],
            "source_urls": list(result.get("source_urls") or []),
            "full_state": final_state,
        }

        for agent in current_state["agent_statuses"]:
            app_state.update_agent_status(agent, "completed")

        run_logger.finish_run(
            symbol=ticker,
            status="completed",
            final_state=final_state,
            final_signal=decision,
        )
        run_started = False
        current_state["chart_data"] = create_chart(ticker, period="1y", end_date=None)
        current_state["analysis_complete"] = True
        if progress is not None:
            progress(1.0)

        trade_enabled = getattr(app_state, 'trade_enabled', False)
        trade_amount = getattr(app_state, 'trade_amount', 1000)
        if trade_enabled:
            print(f"[TRADE] Trading enabled for {ticker}, executing trade with ${trade_amount}")
            execute_trade_after_analysis(ticker, allow_shorts, trade_amount)
        else:
            print(f"[TRADE] Trading disabled for {ticker}, skipping trade execution")
        app_state.needs_ui_update = True

    except Exception as e:
        print(f"Analysis error: {e}")
        import traceback
        traceback.print_exc()
        if run_started:
            run_logger.finish_run(
                symbol=ticker,
                status="failed",
                final_state=final_state,
                error_message=str(e),
            )
            run_started = False
        if current_state:
            current_state["analysis_complete"] = False
            current_state["analysis_error"] = str(e)
            current_state["analysis_results"] = {"error": str(e)}
        if progress is not None:
            progress(1.0)
    finally:
        print(f"Real-time analysis for {ticker} completed")
        if current_state:
            current_state["analysis_running"] = False

    return "Real-time analysis complete"


def start_analysis(
    ticker,
    analysts_market,
    analysts_social,
    analysts_news,
    analysts_fundamentals,
    analysts_macro,
    research_depth,
    allow_shorts,
    quick_llm,
    deep_llm,
    quick_llm_params=None,
    deep_llm_params=None,
    llm_provider="openai",
    backend_url=None,
    output_language="English",
    checkpoint_enabled=False,
    provider_settings=None,
    progress=None,
):
    """Start real-time analysis function for the UI"""

    # Parse selected analysts
    selected_analysts = []
    if analysts_market:
        selected_analysts.append("market")
    if analysts_social:
        selected_analysts.append("social")
    if analysts_news:
        selected_analysts.append("news")
    if analysts_fundamentals:
        selected_analysts.append("fundamentals")
    if analysts_macro:
        selected_analysts.append("macro")

    if not selected_analysts:
        return "Please select at least one analyst type."

    # Convert research depth to integer for debate rounds
    # Also keep the original string for LLM parameter mapping
    if research_depth == "Shallow":
        depth_rounds = 1
    elif research_depth == "Medium":
        depth_rounds = 3
    else:  # Deep
        depth_rounds = 5

    # Pass both the string (for LLM params) and rounds (for debates)
    depth_config = {"rounds": depth_rounds, "level": research_depth}

    # Create an initial chart immediately with current data
    try:
        print(f"Creating initial chart for {ticker} with current market data")
        current_state = app_state.get_state(ticker)
        if current_state:
            current_state["chart_data"] = create_chart(ticker, period="1y", end_date=None)
    except Exception as e:
        print(f"Error creating initial chart: {e}")
        import traceback
        traceback.print_exc()

    # Run analysis with current data
    run_analysis(
        ticker,
        selected_analysts,
        depth_config,
        allow_shorts,
        quick_llm,
        deep_llm,
        quick_llm_params=quick_llm_params,
        deep_llm_params=deep_llm_params,
        llm_provider=llm_provider,
        backend_url=backend_url,
        output_language=output_language,
        checkpoint_enabled=checkpoint_enabled,
        provider_settings=provider_settings,
        progress=progress,
    )

    # Update the status message with more details
    trading_mode = "Trading Mode (LONG/NEUTRAL/SHORT)" if allow_shorts else "Investment Mode (BUY/HOLD/SELL)"
    trade_text = f" with ${getattr(app_state, 'trade_amount', 1000)} optional order execution" if getattr(app_state, 'trade_enabled', False) else ""
    return f"TradingAgents API analysis started for {ticker} with {len(selected_analysts)} analyst selections in {trading_mode}{trade_text}. Alpaca independently validates any order before execution."
