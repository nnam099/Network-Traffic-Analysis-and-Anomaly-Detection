from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable

import numpy as np
import pandas as pd

from .inference_runtime import risk_score, severity_rank, traffic_verdict


Normalizer = Callable[[pd.DataFrame], tuple[pd.DataFrame, Any]]


@dataclass
class DashboardPreprocessResult:
    features: np.ndarray
    normalization_report: dict[str, Any] | None = None


@dataclass
class AIContextOption:
    label: str
    context: dict[str, Any]


def preprocess_dashboard_df(
    df_raw: pd.DataFrame,
    feature_names: list[str],
    model_version: str,
    pipeline_meta: dict[str, Any] | None = None,
    normalizer: Normalizer | None = None,
) -> DashboardPreprocessResult:
    """
    Align dashboard input rows to the exact feature contract expected by the scaler.

    The function is deliberately Streamlit-free so CSV preprocessing can be tested
    without importing the dashboard application.
    """
    if model_version.strip().lower() != "v14":
        raise ValueError(f"unsupported dashboard model version: {model_version}")

    from .dataset import engineer_features

    df = df_raw.copy()
    report_dict: dict[str, Any] | None = None
    if normalizer is not None:
        try:
            df, report = normalizer(df)
            report_dict = report.as_dict() if hasattr(report, "as_dict") else dict(report)
        except Exception as exc:
            report_dict = {
                "schema": "normalization_failed",
                "error": str(exc),
            }

    categorical_maps = (pipeline_meta or {}).get("categorical_maps", {})
    for cat in ["proto", "service", "state"]:
        if cat in df.columns:
            df[f"{cat}_num"] = _encode_categorical_column(df[cat], categorical_maps.get(cat))

    existing_numeric = [
        col for col in df.columns
        if col not in {
            "attack_cat", "label", "label_binary", "srcip", "dstip",
            "sport", "dsport", "stime", "ltime", "id", "proto", "service", "state",
        }
    ]
    try:
        df, _ = engineer_features(df, existing_numeric)
    except Exception:
        pass

    rows = len(df)
    out = np.zeros((rows, len(feature_names)), dtype=np.float32)
    for index, col in enumerate(feature_names):
        if col not in df.columns:
            continue
        try:
            out[:, index] = pd.to_numeric(df[col], errors="coerce").fillna(0).values
        except Exception:
            out[:, index] = 0.0

    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return DashboardPreprocessResult(features=out, normalization_report=report_dict)


def build_alert_context_from_log(
    row_scores: dict[str, Any],
    timestamp: str | None = None,
) -> dict[str, Any]:
    source_row = int(row_scores.get("source_row", 0))
    family = str(row_scores.get("zero_day_family") or "")
    classifier_class = str(row_scores.get("classifier_class", row_scores.get("predicted_class", "Unknown")))
    detection = str(row_scores.get("detection") or traffic_verdict(row_scores.get("is_zeroday"), classifier_class))
    return {
        "alert_id": f"ZD-{source_row:06d}",
        "timestamp": timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_row": source_row,
        "hybrid_score": float(row_scores.get("hybrid_score", 0)),
        "ae_score": float(row_scores.get("ae_score", 0)),
        "max_prob": float(row_scores.get("max_prob", 0)),
        "predicted_class": detection,
        "classifier_class": classifier_class,
        "zero_day_family": family or "",
        "is_zeroday": bool(row_scores.get("is_zeroday", False)),
        "shap_summary": "Batch log context - SHAP explanation not computed for this row.",
        "mitre_summary": "",
        "top_features": [],
        "probs": [],
        "demo_mode": False,
        "raw_scores": {key: str(value) for key, value in row_scores.items()},
    }


def build_ai_context_options(
    alert_history: list[dict[str, Any]] | None = None,
    bulk_logs: pd.DataFrame | None = None,
    max_history: int = 20,
    max_bulk_logs: int = 50,
) -> list[AIContextOption]:
    options: list[AIContextOption] = []

    for item in (alert_history or [])[-max_history:]:
        options.append(AIContextOption(
            label=(
                f"Alert history | {item.get('alert_id')} | {item.get('predicted_class')} | "
                f"score={float(item.get('hybrid_score', 0)):.3g}"
            ),
            context=item,
        ))

    if isinstance(bulk_logs, pd.DataFrame) and not bulk_logs.empty:
        logs_for_ai = bulk_logs.copy()
        if "source_row" not in logs_for_ai.columns:
            logs_for_ai.insert(0, "source_row", np.arange(len(logs_for_ai)))
        sort_cols = [col for col in ["is_zeroday", "hybrid_score", "ae_score"] if col in logs_for_ai.columns]
        if sort_cols:
            logs_for_ai = logs_for_ai.sort_values(sort_cols, ascending=False)
        for _, row in logs_for_ai.head(max_bulk_logs).iterrows():
            row_dict = row.to_dict()
            options.append(AIContextOption(
                label=(
                    f"Zero-day log | row={int(row_dict.get('source_row', 0))} | "
                    f"{row_dict.get('detection', row_dict.get('predicted_class'))} | "
                    f"hybrid={float(row_dict.get('hybrid_score', 0)):.3g}"
                ),
                context=build_alert_context_from_log(row_dict),
            ))

    return options


def filter_alert_history(
    alerts: list[dict[str, Any]],
    status: str = "All",
    severity: str = "All",
    ood_filter: str = "All",
    query: str = "",
) -> list[dict[str, Any]]:
    query = query.strip().lower()
    out: list[dict[str, Any]] = []
    for alert in alerts:
        if status != "All" and str(alert.get("status", "new")) != status:
            continue
        if severity != "All" and str(alert.get("llm_severity", "N/A")).upper() != severity.upper():
            continue
        if ood_filter == "OOD only" and not bool(alert.get("is_zeroday")):
            continue
        if ood_filter == "Known only" and bool(alert.get("is_zeroday")):
            continue
        if query:
            haystack = " ".join(str(alert.get(key, "")) for key in [
                "alert_id", "predicted_class", "classifier_class", "status",
                "source", "zero_day_family", "analyst_note",
            ]).lower()
            if query not in haystack:
                continue
        out.append(alert)
    return out


def build_top_batch_alerts(
    result_df: pd.DataFrame,
    file_hash: str,
    limit: int = 25,
    timestamp: str | None = None,
    raw_df: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(result_df, pd.DataFrame) or result_df.empty or limit <= 0:
        return []

    df = result_df.copy()
    if "source_row" not in df.columns:
        df.insert(0, "source_row", np.arange(len(df)))
    sort_cols = [col for col in ["is_zeroday", "hybrid_score", "ae_score", "max_prob"] if col in df.columns]
    desired_sort = {"is_zeroday": False, "hybrid_score": False, "ae_score": False, "max_prob": True}
    ascending = [desired_sort[col] for col in sort_cols]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=ascending)

    prefix = (file_hash or "unknown")[:8]
    alerts: list[dict[str, Any]] = []
    for _, row in df.head(limit).iterrows():
        row_dict = row.to_dict()
        source_row = int(row_dict.get("source_row", 0))
        alert = build_alert_context_from_log(row_dict, timestamp=timestamp)
        alert["alert_id"] = f"BATCH-{prefix}-{source_row:06d}"
        alert["source_file_hash"] = file_hash
        alert["source"] = "batch_top"
        if isinstance(raw_df, pd.DataFrame) and source_row < len(raw_df):
            alert.update(extract_alert_entities(raw_df.iloc[source_row].to_dict()))
        alerts.append(alert)
    return alerts


def extract_alert_entities(row: dict[str, Any]) -> dict[str, str]:
    return {
        "src_ip": _first_present_value(row, ["srcip", "src_ip", "source_ip", "source.address", "sourceip", "id.orig_h"]),
        "dst_ip": _first_present_value(row, ["dstip", "dst_ip", "destination_ip", "dest_ip", "destination.address", "id.resp_h"]),
        "src_port": _first_present_value(row, ["sport", "src_port", "source_port", "source.port", "id.orig_p"]),
        "dst_port": _first_present_value(row, ["dsport", "dst_port", "destination_port", "dest_port", "destination.port", "id.resp_p"]),
        "protocol": _first_present_value(row, ["proto", "protocol", "transport", "network.transport"]),
        "service": _first_present_value(row, ["service", "app_proto", "application", "app", "protocol_name"]),
    }


def correlate_alerts(alerts: list[dict[str, Any]], min_count: int = 2) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for alert in alerts:
        for group_type, key in _correlation_keys(alert):
            groups.setdefault((group_type, key), []).append(alert)

    out: list[dict[str, Any]] = []
    for (group_type, key), members in groups.items():
        if len(members) < min_count:
            continue
        out.append({
            "group_type": group_type,
            "key": key,
            "alert_count": len(members),
            "ood_count": sum(1 for item in members if bool(item.get("is_zeroday"))),
            "max_risk": max(_safe_float(item.get("risk"), default=0.0) for item in members),
            "latest_time": max(str(item.get("timestamp", "")) for item in members),
            "alert_ids": [str(item.get("alert_id", "")) for item in members[:10]],
        })

    return sorted(out, key=lambda item: (item["alert_count"], item["ood_count"], item["max_risk"]), reverse=True)


def build_time_window_incidents(
    alerts: list[dict[str, Any]],
    window_minutes: int = 15,
    min_alerts: int = 2,
    max_alert_ids: int = 12,
) -> list[dict[str, Any]]:
    """Group repeated correlated alerts into analyst-friendly incident windows."""
    if window_minutes <= 0 or min_alerts <= 0:
        return []

    keyed_events: dict[tuple[str, str], list[tuple[datetime, dict[str, Any]]]] = {}
    for alert in alerts:
        event_time = _parse_alert_time(alert)
        if event_time is None:
            continue
        for group_type, key in _correlation_keys(alert):
            if group_type == "Batch File":
                continue
            keyed_events.setdefault((group_type, key), []).append((event_time, alert))

    window = timedelta(minutes=window_minutes)
    incidents: list[dict[str, Any]] = []
    for (group_type, key), events in keyed_events.items():
        events = sorted(events, key=lambda item: item[0])
        cluster: list[tuple[datetime, dict[str, Any]]] = []
        cluster_start: datetime | None = None
        for event_time, alert in events:
            if cluster_start is None or event_time - cluster_start <= window:
                if cluster_start is None:
                    cluster_start = event_time
                cluster.append((event_time, alert))
                continue
            incidents.extend(
                _incident_from_cluster(group_type, key, cluster, window_minutes, min_alerts, max_alert_ids)
            )
            cluster = [(event_time, alert)]
            cluster_start = event_time
        incidents.extend(
            _incident_from_cluster(group_type, key, cluster, window_minutes, min_alerts, max_alert_ids)
        )

    return sorted(
        incidents,
        key=lambda item: (item["max_risk"], item["ood_count"], item["alert_count"], item["end_time"]),
        reverse=True,
    )


def default_ai_context_index(options: list[AIContextOption], current_alert_id: str | None) -> int:
    for index, option in enumerate(options):
        if option.context.get("alert_id") == current_alert_id:
            return index
    return 0


def triage_alert_with_fallback(result: dict[str, Any], agent: Any | None = None) -> dict[str, Any]:
    if agent is None:
        return {
            "severity": "HIGH" if result["hybrid_score"] > 0.6 else "MEDIUM",
            "verdict": (
                f"{'Zero-Day Candidate' if result['is_zeroday'] else result['predicted_class']} "
                f"detected - hybrid score: {result['hybrid_score']:.3f}"
            ),
            "attack_summary": (
                f"AE reconstruction error cao ({result['ae_score']:.3f}) cho thay traffic bat thuong. "
                "Can kiem tra thu cong."
            ),
            "recommended_actions": ["Kiem tra nguon IP", "Xem xet block traffic", "Escalate len Tier 2"],
            "false_positive_risk": "MEDIUM",
            "false_positive_reason": "Chua co LLM de phan tich sau hon",
            "analyst_note": "Kich hoat LLM agent de co phan tich chi tiet hon",
        }
    try:
        return agent.triage_alert(result)
    except Exception as exc:
        return {
            "severity": "HIGH",
            "verdict": "LLM analysis loi - can review thu cong",
            "attack_summary": str(exc),
            "recommended_actions": ["Review thu cong"],
            "false_positive_risk": "UNKNOWN",
            "false_positive_reason": "N/A",
            "analyst_note": f"Loi: {exc}",
        }


def answer_analyst_question(
    question: str,
    alert_context: dict[str, Any],
    has_llm: bool,
    agent_factory: Callable[[], Any] | None = None,
    llm_provider: str | None = None,
    llm_dependency: str | None = None,
    has_llm_dependency: bool = True,
) -> str:
    if has_llm and agent_factory is not None:
        try:
            return str(agent_factory().explain_to_analyst(question, alert_context))
        except Exception as exc:
            return f"Loi LLM Agent: {exc}. Vui long kiem tra lai provider va API Key trong file .env."

    if llm_dependency and not has_llm_dependency:
        return f"Chua cai thu vien cho LLM provider '{llm_provider}'. Can cai: {llm_dependency}."
    return "Khong tim thay llm_agent.py. Vui long kiem tra lai source code."


def _encode_categorical_column(series: pd.Series, mapping=None) -> np.ndarray:
    values = series.astype(str).fillna("unk")
    if mapping:
        return values.map(lambda x: mapping.get(x, mapping.get("unk", -1))).astype(np.float32).values
    codes, _ = pd.factorize(values, sort=True)
    return codes.astype(np.float32)


def _first_present_value(row: dict[str, Any], names: list[str]) -> str:
    lower = {str(key).strip().lower(): value for key, value in row.items()}
    for name in names:
        value = lower.get(name.lower())
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none", "null", "-", "unknown"}:
            return text
    return ""


def _correlation_keys(alert: dict[str, Any]) -> list[tuple[str, str]]:
    candidates = [
        ("Source IP", alert.get("src_ip")),
        ("Destination IP", alert.get("dst_ip")),
        ("Service", alert.get("service")),
        ("Classifier Class", alert.get("classifier_class")),
        ("Zero-Day Family", alert.get("zero_day_family")),
        ("Batch File", alert.get("source_file_hash")),
    ]
    out = []
    for group_type, value in candidates:
        text = str(value or "").strip()
        if text and text.lower() not in {"normal", "nan", "none", "null", "-", "unknown"}:
            out.append((group_type, text))
    return out


def _incident_from_cluster(
    group_type: str,
    key: str,
    cluster: list[tuple[datetime, dict[str, Any]]],
    window_minutes: int,
    min_alerts: int,
    max_alert_ids: int,
) -> list[dict[str, Any]]:
    if len(cluster) < min_alerts:
        return []

    start = min(item[0] for item in cluster)
    end = max(item[0] for item in cluster)
    alerts = [item[1] for item in cluster]
    risks = [_alert_risk(alert) for alert in alerts]
    max_risk = max(risks) if risks else 0
    alert_ids = [str(alert.get("alert_id", "")) for alert in alerts if str(alert.get("alert_id", "")).strip()]
    classes = _top_values(alerts, "classifier_class", fallback_key="predicted_class")
    families = _top_values(alerts, "zero_day_family")
    source_ips = _top_values(alerts, "src_ip")
    services = _top_values(alerts, "service")
    severity_values = [str(alert.get("llm_severity", "")).upper() for alert in alerts]
    strongest_severity = max(severity_values, key=severity_rank, default="")
    severity = strongest_severity if severity_rank(strongest_severity) else _risk_to_severity(max_risk)
    raw_id = f"{group_type}|{key}|{start.isoformat()}|{end.isoformat()}"
    incident_id = "INC-" + hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:10].upper()

    return [{
        "incident_id": incident_id,
        "group_type": group_type,
        "key": key,
        "window_minutes": window_minutes,
        "start_time": _format_incident_time(start),
        "end_time": _format_incident_time(end),
        "duration_minutes": round(max(0.0, (end - start).total_seconds() / 60), 1),
        "alert_count": len(alerts),
        "ood_count": sum(1 for alert in alerts if bool(alert.get("is_zeroday"))),
        "high_count": sum(1 for alert in alerts if _alert_risk(alert) >= 70),
        "max_risk": max_risk,
        "severity": severity,
        "primary_classes": classes,
        "families": families,
        "source_ips": source_ips,
        "services": services,
        "alert_ids": alert_ids[:max_alert_ids],
        "recommended_focus": _incident_focus(group_type, key, classes, families),
    }]


def _parse_alert_time(alert: dict[str, Any]) -> datetime | None:
    for key in ["timestamp", "event_timestamp", "created_at", "updated_at"]:
        value = alert.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        normalized = text.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
            return parsed.replace(tzinfo=None)
        except ValueError:
            pass
        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%d/%m/%Y %H:%M:%S"]:
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
    return None


def _format_incident_time(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _alert_risk(alert: dict[str, Any]) -> int:
    if alert.get("risk") is not None:
        return int(_safe_float(alert.get("risk"), default=0.0))
    return risk_score(alert, str(alert.get("llm_severity") or alert.get("severity") or ""))


def _risk_to_severity(risk: float) -> str:
    if risk >= 85:
        return "CRITICAL"
    if risk >= 70:
        return "HIGH"
    if risk >= 40:
        return "MEDIUM"
    return "LOW"


def _top_values(
    alerts: list[dict[str, Any]],
    key: str,
    fallback_key: str | None = None,
    limit: int = 3,
) -> list[str]:
    counts: dict[str, int] = {}
    for alert in alerts:
        value = alert.get(key)
        if (value is None or str(value).strip().lower() in {"", "nan", "none", "null", "-", "unknown"}) and fallback_key:
            value = alert.get(fallback_key)
        text = str(value or "").strip()
        if text and text.lower() not in {"normal", "nan", "none", "null", "-", "unknown"}:
            counts[text] = counts.get(text, 0) + 1
    return [item[0] for item in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def _incident_focus(group_type: str, key: str, classes: list[str], families: list[str]) -> str:
    subject = f"{group_type}: {key}"
    signals = classes or families
    if signals:
        return f"Review repeated alerts for {subject}; dominant signal: {', '.join(signals[:2])}."
    return f"Review repeated alerts for {subject} within the same time window."


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
