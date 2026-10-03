"""Dedicated historical-validation worker; isolated from paper execution."""
import fcntl
import logging
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

from .clients import PaperBroker, ResearchClient
from .store import Store
from .validation import HistoricalValidator


def tick(store, validator):
    store.set("validation_heartbeat", datetime.now(timezone.utc).isoformat())
    for row in reversed(store.validations(20)):
        if row["status"] == "queued":
            validator.run(row["id"])
            break


def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    store = Store()
    with open(str(store.path) + ".validation.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another historical-validation worker owns this database") from None
        for row in store.validations(20):
            if row["status"] == "running":
                payload = row["payload"]
                if payload.get("stop_requested"):
                    payload["progress"].update(
                        step="stopped", message="Validation stopped; completed results were kept"
                    )
                    store.update_validation(row["id"], "stopped", payload)
                else:
                    payload["error"] = "worker_restarted"
                    payload["progress"].update(
                        step="failed", message="Validation was interrupted by a worker restart"
                    )
                    store.update_validation(row["id"], "failed", payload)
        validator = HistoricalValidator(store, PaperBroker(), ResearchClient())
        while True:
            try:
                tick(store, validator)
                store.set("validation_worker_error", None)
            except Exception as exc:
                store.set("validation_worker_error", type(exc).__name__)
                logging.error("Validation worker tick failed: %s", type(exc).__name__)
            time.sleep(5)


if __name__ == "__main__":
    main()
