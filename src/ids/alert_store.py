from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from typing import Any


SCHEMA_VERSION = 1


def init_alert_store(db_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                event_timestamp TEXT,
                source TEXT NOT NULL DEFAULT 'single',
                status TEXT NOT NULL DEFAULT 'new',
                severity TEXT,
                risk INTEGER,
                predicted_class TEXT,
                classifier_class TEXT,
                is_zeroday INTEGER NOT NULL DEFAULT 0,
                hybrid_score REAL,
                ae_score REAL,
                max_prob REAL,
                analyst_note TEXT,
                alert_json TEXT NOT NULL,
                llm_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS alert_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT,
                FOREIGN KEY(alert_id) REFERENCES alerts(alert_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS store_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT OR REPLACE INTO store_meta(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        conn.commit()


def save_alert(
    db_path: str,
    alert: dict[str, Any],
    llm: dict[str, Any] | None = None,
    source: str = "single",
    status: str = "new",
) -> None:
    init_alert_store(db_path)
    now = _utc_now()
    alert_id = str(alert.get("alert_id") or "").strip()
    if not alert_id:
        raise ValueError("alert must include alert_id")

    severity = str((llm or {}).get("severity") or alert.get("llm_severity") or "")
    risk = _safe_int(alert.get("risk"))
    row = (
        alert_id,
        now,
        now,
        str(alert.get("timestamp") or ""),
        source,
        status,
        severity or None,
        risk,
        str(alert.get("predicted_class") or ""),
        str(alert.get("classifier_class") or alert.get("predicted_class") or ""),
        1 if bool(alert.get("is_zeroday")) else 0,
        _safe_float(alert.get("hybrid_score")),
        _safe_float(alert.get("ae_score")),
        _safe_float(alert.get("max_prob")),
        str((llm or {}).get("analyst_note") or alert.get("analyst_note") or ""),
        _json_dumps(alert),
        _json_dumps(llm) if llm is not None else None,
    )

    with closing(sqlite3.connect(db_path)) as conn:
        existing = conn.execute("SELECT created_at FROM alerts WHERE alert_id = ?", (alert_id,)).fetchone()
        created_at = existing[0] if existing else now
        conn.execute(
            """
            INSERT INTO alerts (
                alert_id, created_at, updated_at, event_timestamp, source, status,
                severity, risk, predicted_class, classifier_class, is_zeroday,
                hybrid_score, ae_score, max_prob, analyst_note, alert_json, llm_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(alert_id) DO UPDATE SET
                updated_at=excluded.updated_at,
                event_timestamp=excluded.event_timestamp,
                source=excluded.source,
                status=excluded.status,
                severity=excluded.severity,
                risk=excluded.risk,
                predicted_class=excluded.predicted_class,
                classifier_class=excluded.classifier_class,
                is_zeroday=excluded.is_zeroday,
                hybrid_score=excluded.hybrid_score,
                ae_score=excluded.ae_score,
                max_prob=excluded.max_prob,
                analyst_note=excluded.analyst_note,
                alert_json=excluded.alert_json,
                llm_json=excluded.llm_json
            """,
            (alert_id, created_at, *row[2:]),
        )
        conn.execute(
            """
            INSERT INTO alert_events(alert_id, created_at, event_type, payload_json)
            VALUES (?, ?, ?, ?)
            """,
            (alert_id, now, "saved", _json_dumps({"source": source, "status": status, "severity": severity})),
        )
        conn.commit()


def list_alerts(db_path: str, limit: int = 200, status: str | None = None) -> list[dict[str, Any]]:
    if not os.path.exists(db_path):
        return []
    init_alert_store(db_path)
    query = """
        SELECT alert_json, llm_json, status, severity, risk, analyst_note, source
        FROM alerts
    """
    params: list[Any] = []
    if status:
        query += " WHERE status = ?"
        params.append(status)
    query += " ORDER BY updated_at DESC LIMIT ?"
    params.append(int(limit))

    out: list[dict[str, Any]] = []
    with closing(sqlite3.connect(db_path)) as conn:
        for alert_json, llm_json, row_status, severity, risk, note, source in conn.execute(query, params):
            alert = _json_loads(alert_json)
            llm = _json_loads(llm_json) if llm_json else {}
            alert["status"] = row_status
            alert["source"] = source
            if severity:
                alert["llm_severity"] = severity
            if risk is not None:
                alert["risk"] = int(risk)
            if note:
                alert["analyst_note"] = note
            if llm:
                alert["llm_analysis"] = llm
            out.append(alert)
    return out


def update_alert_status(db_path: str, alert_id: str, status: str, analyst_note: str = "") -> None:
    init_alert_store(db_path)
    now = _utc_now()
    alert_id = str(alert_id or "").strip()
    if not alert_id:
        raise ValueError("alert_id must not be empty")

    with closing(sqlite3.connect(db_path)) as conn:
        cursor = conn.execute(
            "UPDATE alerts SET status = ?, analyst_note = ?, updated_at = ? WHERE alert_id = ?",
            (status, analyst_note, now, alert_id),
        )
        if cursor.rowcount == 0:
            raise ValueError(f"alert not found: {alert_id}")
        conn.execute(
            """
            INSERT INTO alert_events(alert_id, created_at, event_type, payload_json)
            VALUES (?, ?, ?, ?)
            """,
            (alert_id, now, "status_updated", _json_dumps({"status": status, "analyst_note": analyst_note})),
        )
        conn.commit()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _json_loads(value: str) -> dict[str, Any]:
    loaded = json.loads(value)
    return loaded if isinstance(loaded, dict) else {}


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
