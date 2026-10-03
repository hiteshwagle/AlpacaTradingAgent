"""Transactional single-host journal shared by the dashboard and one worker.

SQLite WAL lives on a local persistent volume, never on an NFS mount. The worker
holds an OS lock for its lifetime; restart reconciliation never resubmits orders.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from .models import Policy, next_run, utcnow


def encode(value):
    return json.dumps(value, allow_nan=False, sort_keys=True)


class Store:
    def __init__(self, path=None):
        self.path = Path(path or os.getenv("PORTFOLIO_DB_PATH", ".tradingagents/portfolio.sqlite3"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.db() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS cycles (
                    id TEXT PRIMARY KEY, trigger_key TEXT UNIQUE, status TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE UNIQUE INDEX IF NOT EXISTS one_active_cycle ON cycles ((1))
                    WHERE status IN ('queued','researching','executing','reconciling','needs_review');
                CREATE TABLE IF NOT EXISTS orders (
                    client_id TEXT PRIMARY KEY, cycle_id TEXT NOT NULL,
                    state TEXT NOT NULL, request TEXT NOT NULL, response TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, cycle_id TEXT,
                    time TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, time TEXT NOT NULL,
                    account_id TEXT NOT NULL, equity REAL NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS validation_runs (
                    id TEXT PRIMARY KEY, status TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    request TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE UNIQUE INDEX IF NOT EXISTS one_active_validation ON validation_runs ((1))
                    WHERE status IN ('queued','running');
            """)

    @contextmanager
    def db(self):
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def get(self, key, default=None):
        with self.db() as db:
            row = db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set(self, key, value):
        with self.db() as db:
            db.execute("INSERT OR REPLACE INTO settings VALUES (?,?)", (key, encode(value)))

    def policy(self):
        return Policy.model_validate(self.get("policy", {}))

    def save_policy(self, policy):
        with self.db() as db:
            for key, value in (("policy", policy.model_dump()), ("next_run", next_run(policy, utcnow()))):
                db.execute("INSERT OR REPLACE INTO settings VALUES (?,?)", (key, encode(value)))

    def enqueue(self, *, trigger_key=None, policy=None):
        if self.get("halted", False):
            raise ValueError("Portfolio emergency stop is active")
        cycle_id = uuid4().hex
        now = utcnow().isoformat()
        payload = {"policy": (policy or self.policy()).model_dump(), "progress": {
            "step": "queued", "message": "Waiting for the portfolio worker",
            "percent": 5, "completed": 0, "total": 0,
        }}
        try:
            with self.db() as db:
                db.execute("INSERT INTO cycles VALUES (?,?,?,?,?,?)",
                           (cycle_id, trigger_key, "queued", now, now, encode(payload)))
        except sqlite3.IntegrityError:
            raise ValueError("A portfolio cycle already exists or awaits reconciliation") from None
        self.event(cycle_id, "queued", payload)
        return cycle_id

    def event(self, cycle_id, kind, payload):
        with self.db() as db:
            db.execute("INSERT INTO events(cycle_id,time,kind,payload) VALUES (?,?,?,?)",
                       (cycle_id, utcnow().isoformat(), kind, encode(payload)))

    def update(self, cycle_id, status, payload):
        with self.db() as db:
            db.execute("UPDATE cycles SET status=?,updated_at=?,payload=? WHERE id=?",
                       (status, utcnow().isoformat(), encode(payload), cycle_id))
        self.event(cycle_id, status, {"error": payload.get("error")})

    def cycles(self, limit=50):
        with self.db() as db:
            rows = db.execute("SELECT * FROM cycles ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [{**dict(r), "payload": json.loads(r["payload"])} for r in rows]

    def cycle(self, cycle_id):
        with self.db() as db:
            row = db.execute("SELECT * FROM cycles WHERE id=?", (cycle_id,)).fetchone()
        return {**dict(row), "payload": json.loads(row["payload"])} if row else None

    def order_intent(self, cycle_id, client_id, request):
        with self.db() as db:
            db.execute("INSERT INTO orders VALUES (?,?,?,?,?)",
                       (client_id, cycle_id, "submitting", encode(request), "{}"))

    def order_result(self, client_id, response):
        with self.db() as db:
            db.execute("UPDATE orders SET state=?,response=? WHERE client_id=?",
                       (response.get("status", "unknown"), encode(response), client_id))

    def orders(self, cycle_id=None):
        with self.db() as db:
            rows = db.execute("SELECT * FROM orders" + (" WHERE cycle_id=?" if cycle_id else ""),
                              (cycle_id,) if cycle_id else ()).fetchall()
        return [{**dict(r), "request": json.loads(r["request"]), "response": json.loads(r["response"])} for r in rows]

    def evaluate(self, account, summary):
        with self.db() as db:
            db.execute("INSERT INTO evaluations(time,account_id,equity,payload) VALUES (?,?,?,?)",
                       (utcnow().isoformat(), account["id"], float(account["equity"]), encode(summary)))

    def evaluations(self, account_id):
        with self.db() as db:
            rows = db.execute("SELECT * FROM evaluations WHERE account_id=? ORDER BY id DESC LIMIT 100",
                              (account_id,)).fetchall()
        return [{**dict(r), "payload": json.loads(r["payload"])} for r in rows]

    def enqueue_validation(self, request):
        run_id = "val_" + uuid4().hex
        now = utcnow().isoformat()
        total = len(request.symbols) * len(request.analysis_dates)
        payload = {"progress": {"step": "queued", "message": "Waiting for the validation worker",
                                "percent": 0, "completed": 0, "total": total},
                   "results": [], "summary": {}}
        try:
            with self.db() as db:
                db.execute("INSERT INTO validation_runs VALUES (?,?,?,?,?,?)",
                           (run_id, "queued", now, now, encode(request.model_dump(mode="json")),
                            encode(payload)))
        except sqlite3.IntegrityError:
            raise ValueError("A historical validation is already running") from None
        return run_id

    def update_validation(self, run_id, status, payload):
        with self.db() as db:
            db.execute("UPDATE validation_runs SET status=?,updated_at=?,payload=? WHERE id=?",
                       (status, utcnow().isoformat(), encode(payload), run_id))

    def validations(self, limit=20):
        with self.db() as db:
            rows = db.execute(
                "SELECT * FROM validation_runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [{**dict(row), "request": json.loads(row["request"]),
                 "payload": json.loads(row["payload"])} for row in rows]

    def validation(self, run_id):
        with self.db() as db:
            row = db.execute("SELECT * FROM validation_runs WHERE id=?", (run_id,)).fetchone()
        return ({**dict(row), "request": json.loads(row["request"]),
                 "payload": json.loads(row["payload"])}) if row else None
