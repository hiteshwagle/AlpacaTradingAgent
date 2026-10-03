"""Run with python -m portfolio_service.worker; one process per database."""
import fcntl
import logging
import time

from dotenv import load_dotenv

from .clients import PaperBroker, ResearchClient, ScreenerClient
from .engine import Engine
from .models import next_run, timestamp, utcnow
from .store import Store

logger = logging.getLogger(__name__)


def tick(store, engine):
    store.set("heartbeat", utcnow().isoformat())
    for cycle in reversed(store.cycles(100)):
        if cycle["status"] in {"researching", "executing", "reconciling", "needs_review"}:
            engine.reconcile(cycle["id"])
    policy = store.policy()
    due = store.get("next_run")
    if due and policy.schedule_enabled and timestamp(due) <= utcnow():
        if not store.get("halted", False) and engine.broker.clock().get("is_open"):
            try:
                store.enqueue(trigger_key="schedule:" + due, policy=policy)
            except ValueError:
                pass  # Coalesce missed/overlapping schedules, never backlog trades.
        store.set("next_run", next_run(policy, utcnow()))
    if not store.get("halted", False):
        for cycle in reversed(store.cycles(100)):
            if cycle["status"] == "queued":
                engine.run(cycle["id"])
                break


def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    store = Store()
    with open(str(store.path) + ".worker.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another portfolio worker owns this database") from None
        engine = Engine(store, PaperBroker(), ResearchClient(), ScreenerClient())
        while True:
            try:
                tick(store, engine)
                store.set("worker_error", None)
            except Exception as exc:
                # Do not leak provider URLs, response bodies, or credentials.
                store.set("worker_error", type(exc).__name__)
                logger.error("Worker tick failed: %s", type(exc).__name__)
            time.sleep(5)


if __name__ == "__main__":
    main()
