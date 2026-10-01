"""Build fail-closed P11 artifacts without touching P9 checkpoints or results."""

from __future__ import annotations

from collections import Counter
import hashlib
from importlib import metadata
import json
import platform
from pathlib import Path
import sys

from .categorical_features import application_candidates, protocol_token, state_candidate
from .continuous_features import add_frozen_engineered_features, extract_candidate_features
from .flow_builder import FlowBuilder
from .packet_reader import PacketRecord
from .schema import VERSION, build_schema_contract, semantic_inventory
from .serialization import serialized_sha256
from .validation import declared_tolerance


P8_PROTOCOL_SHA256 = "62916073ef6048503415d45e2605bbf475b729e032c838a60cc7e94c43768216"
P9_PROTOCOL_SHA256 = "c1c7352a9ca546a3e8e456b112d09fd42f29243e205a8760703b27f68b5baf87"
P9_FREEZE_COMMIT = "3dd5f12a6bf3948efff817654111347d1fedc7ce"
P10_STATUS = "P10_EXTERNAL_SOURCE_BLOCKED"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes(root: Path) -> dict[str, str]:
    base = root / "src/ids/p11_extractor"
    paths = sorted(path for path in base.glob("*.py"))
    paths.append(root / "scripts/audit_p11_extractor.py")
    return {str(path.relative_to(root)): file_sha256(path) for path in paths}


def verify_upstream(root: Path) -> dict:
    schema = json.loads((root / "results/data_quality/p3/schema_v2.json").read_text())
    p9 = json.loads((root / "results/loafo/p9/p9_matrix.json").read_text())
    p10 = json.loads((root / "results/loafo/p10/p10_preregistration.json").read_text())
    if schema["sha256"] != "f3ecf12891e5b0d9b5e1fe38a22c23720a71eaf1d5a47b763911df363b3b36d7":
        raise ValueError("frozen 61-field schema mismatch")
    if (p9.get("status") != "P9_DEVELOPMENT_PASS" or
            p9.get("protocol_sha256") != P9_PROTOCOL_SHA256 or
            p9.get("p8_protocol_sha256") != P8_PROTOCOL_SHA256):
        raise ValueError("P8/P9 binding mismatch")
    if (p10.get("status") != P10_STATUS or p10.get("p9_model_evaluated") is not False or
            p10.get("external_source_reserved") is not False):
        raise ValueError("P10A fail-closed boundary mismatch")
    return schema


def original_lineage() -> dict:
    evidence = "https://research.unsw.edu.au/projects/unsw-nb15-dataset"
    paper = "https://doi.org/10.1109/MilCIS.2015.7348942"
    return {
        "version": VERSION, "overall_confidence": "PARTIAL",
        "components": [
            {"component": "traffic generation", "finding": "IXIA PerfectStorm hybrid normal/attack traffic",
             "confidence": "HIGH", "evidence": [evidence, paper]},
            {"component": "packet capture", "finding": "tcpdump; about 100 GB PCAP",
             "confidence": "HIGH", "evidence": [evidence, paper]},
            {"component": "flow engine", "finding": "Argus server/client generated binary Argus and flow fields",
             "confidence": "HIGH_TOOL_LOW_CONFIGURATION", "evidence": [evidence, paper],
             "unresolved": ["Argus version", "command line", "timeouts", "direction policy", "metric modes"]},
            {"component": "application analysis", "finding": "Bro-IDS conn/http/ftp logs",
             "confidence": "HIGH_TOOL_LOW_CONFIGURATION", "evidence": [paper],
             "unresolved": ["Bro version", "scripts", "service analyzer policy", "log rotation"]},
            {"component": "merge", "finding": "Argus and Bro outputs loaded into SQL Server 2008 and matched by flow features",
             "confidence": "MEDIUM", "evidence": [paper],
             "unresolved": ["join keys", "tie handling", "one-to-many handling", "clock normalization"]},
            {"component": "post-processing", "finding": "twelve C# algorithms generated additional/context fields",
             "confidence": "HIGH_EXISTENCE_LOW_REPRODUCIBILITY", "evidence": [evidence, paper],
             "unresolved": ["source code", "exact last-100 ordering", "capture boundaries", "numeric types"]},
            {"component": "labels", "finding": "IXIA event ground truth attached using GT and LIST_EVENTS artifacts",
             "confidence": "MEDIUM", "evidence": [evidence, paper],
             "unresolved": ["exact row association implementation"]},
        ],
        "exact_reproduction_blockers": [
            "historical Argus/Bro versions and configs are not frozen",
            "twelve C# algorithm implementations are not present",
            "row-level raw-PCAP to expected-CSV association has not been validated",
        ],
    }


def reference_corpus(root: Path) -> dict:
    local_raw = sorted(str(path.relative_to(root)) for path in (root / "data").glob("*.pcap*"))
    local_expected = [str(path.relative_to(root)) for path in (
        root / "data/UNSW_NB15_training-set.csv", root / "data/UNSW_NB15_testing-set.csv") if path.exists()]
    return {
        "version": VERSION, "status": "P11_REFERENCE_CORPUS_BLOCKED",
        "candidate": "official full UNSW-NB15 source bundle",
        "authoritative_source": "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
        "documented_bundle": ["PCAP", "Argus files", "Bro files", "UNSW-NB15_1.csv through _4.csv",
                              "UNSW-NB15_GT.csv", "UNSW-NB15_LIST_EVENTS.csv"],
        "availability_documented": True, "local_raw_capture_files": local_raw,
        "local_expected_feature_files": local_expected,
        "local_pair_is_row_alignable": False,
        "raw_capture_sha256": {}, "expected_rows_sha256": {},
        "download_attempt": "official SharePoint link redirected to Microsoft authentication",
        "blocking_reasons": [
            "no authoritative raw PCAP is locally materialized and hashed",
            "local train/test CSV partitions omit IP/time fields required for PCAP-flow row alignment",
            "no immutable mapping from raw packets to the local expected rows is available",
        ],
        "distribution_only_validation_forbidden": True,
        "external_evaluation_pcap_reserved": False,
    }


def _reproduction(field: dict) -> tuple[str, str]:
    name, category = field["name"], field["semantic_class"]
    if name in {"service", "state"}:
        return "NOT_REPRODUCIBLE", "historical analyzer/state-machine mapping and configuration unavailable"
    if category == "LAST_N_CONNECTION_CONTEXT":
        return "NOT_REPRODUCIBLE", "original C# last-N algorithm and boundary/tie rules unavailable"
    if category == "APPLICATION_PROTOCOL_DERIVED":
        return "APPROXIMATE", "local payload parser is only a candidate and is not Bro-version equivalent"
    if name in {"sloss", "dloss", "loss_rate_src", "loss_rate_dst"}:
        return "NOT_REPRODUCIBLE", "packet loss cannot be inferred faithfully from an arbitrary capture point"
    return "APPROXIMATE", "candidate computation exists but has no row-wise authoritative reference comparison"


def validation_matrix(contract: dict) -> dict:
    rows = []
    for field in contract["fields"]:
        reproduction, reason = _reproduction(field)
        tolerance = declared_tolerance(field["units"])
        rows.append({
            "feature": field["name"], "category": field["semantic_class"],
            "reference_support": False, "reproduction_class": reproduction,
            "error": None,
            "tolerance": {"absolute": tolerance.absolute, "relative": tolerance.relative,
                          "rationale": tolerance.rationale},
            "gate": "BLOCK", "reason": reason,
        })
    return {"version": VERSION, "row_count": len(rows), "rows": rows,
            "reproduction_counts": dict(sorted(Counter(row["reproduction_class"] for row in rows).items())),
            "all_61_pass": False}


def continuous_validation(contract: dict, matrix: dict) -> dict:
    by_name = {row["feature"]: row for row in matrix["rows"]}
    features = []
    for field in contract["fields"]:
        if field["type"] == "UTF-8 categorical token":
            continue
        row = by_name[field["name"]]
        features.append({
            "feature": field["name"], "reference_row_count": 0, "exact_match_count": 0,
            "absolute_error": None, "relative_error": None, "maximum_error": None,
            "error_quantiles": None, "nan_mismatch_count": None, "inf_mismatch_count": None,
            "sign_mismatch_count": None, "tolerance": row["tolerance"],
            "reproduction_class": row["reproduction_class"], "validated": False,
        })
    return {"version": VERSION, "comparison_stage": "pre-RobustScaler float64",
            "float32_used": False, "reference_corpus_available": False, "features": features}


def categorical_validation() -> dict:
    return {
        "version": VERSION, "reference_corpus_available": False,
        "features": {
            "proto": {"candidate": "IANA protocol number to lowercase token",
                      "reproduction_class": "APPROXIMATE", "normalization_validated": False},
            "service": {"candidate": "payload-evidenced HTTP/FTP only; never destination-port lookup",
                        "reproduction_class": "NOT_REPRODUCIBLE", "historical_bro_mapping_validated": False},
            "state": {"candidate": "LOCAL_* TCP summary kept outside P9 token namespace",
                      "reproduction_class": "NOT_REPRODUCIBLE", "argus_transition_mapping_validated": False,
                      "zeek_conn_state_substitution_forbidden": True},
        },
    }


def deterministic_probe() -> dict:
    packets = [
        PacketRecord(2, 1_200_000_000, 4, "10.0.0.2", "10.0.0.1", 6, 80, 1234, 40, 63,
                     tcp_flags=0x12, tcp_seq=900, tcp_ack=101, tcp_window=2048),
        PacketRecord(0, 1_000_000_000, 4, "10.0.0.1", "10.0.0.2", 6, 1234, 80, 40, 64,
                     tcp_flags=0x02, tcp_seq=100, tcp_ack=0, tcp_window=4096),
        PacketRecord(3, 1_300_000_000, 4, "10.0.0.1", "10.0.0.2", 6, 1234, 80, 40, 64,
                     tcp_flags=0x10, tcp_seq=101, tcp_ack=901, tcp_window=4096),
    ]
    builder = FlowBuilder()
    outputs = []
    for supplied in (packets, list(reversed(packets))):
        flow = builder.build(supplied)[0]
        outputs.append(add_frozen_engineered_features(extract_candidate_features(flow)))
    hashes = [serialized_sha256(value) for value in outputs]
    return {"version": VERSION, "probe": "synthetic three-way handshake in two input orders",
            "runs": 2, "output_sha256": hashes, "content_equal": outputs[0] == outputs[1],
            "pass": hashes[0] == hashes[1] and outputs[0] == outputs[1],
            "scope": "synthetic candidate extractor only; not reference semantic validation"}


def environment(source: dict[str, str]) -> dict:
    distributions = {}
    for name in ("pytest", "ruff"):
        try:
            distributions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            distributions[name] = None
    return {
        "version": VERSION, "python": sys.version, "implementation": platform.python_implementation(),
        "platform": platform.platform(), "byteorder": sys.byteorder,
        "extractor_runtime_dependencies": "Python standard library only",
        "development_tool_versions": distributions,
        "external_binaries": {}, "historical_argus_version": None, "historical_bro_version": None,
        "source_code_sha256": source,
    }


def gap_analysis(matrix: dict) -> dict:
    groups: dict[str, list[str]] = {}
    for row in matrix["rows"]:
        groups.setdefault(row["reproduction_class"], []).append(row["feature"])
    portable = [row["feature"] for row in matrix["rows"]
                if row["category"] in {"PACKET_DIRECT", "FLOW_DERIVED"} and
                row["reproduction_class"] == "APPROXIMATE"]
    return {
        "version": VERSION, "exact_p9_compatibility": False,
        "features_by_reproduction_class": groups,
        "candidate_portable_subset": portable,
        "candidate_portable_subset_is_approved": False,
        "impact": "P9 accepts exactly 61 frozen semantics; approximate/missing fields prohibit P9 evaluation.",
        "reduced_model_training_performed": False,
        "next_protocol": "separate portable-schema amendment with new model training, never P11",
    }


def build_artifacts(root: Path, created_at_utc: str) -> dict[str, dict]:
    frozen = verify_upstream(root)
    hashes = source_hashes(root)
    contract = build_schema_contract(frozen, hashes)
    inventory = semantic_inventory(contract)
    matrix = validation_matrix(contract)
    corpus = reference_corpus(root)
    artifacts = {
        "schema_contract.json": contract,
        "feature_semantic_inventory.json": inventory,
        "original_extractor_lineage.json": original_lineage(),
        "reference_corpus.json": corpus,
        "extractor_environment.json": environment(hashes),
        "feature_validation_matrix.json": matrix,
        "categorical_validation.json": categorical_validation(),
        "continuous_validation.json": continuous_validation(contract, matrix),
        "determinism_validation.json": deterministic_probe(),
        "portable_schema_gap_analysis.json": gap_analysis(matrix),
    }
    artifact_hashes = {name: hashlib.sha256((json.dumps(value, indent=2, sort_keys=True,
        ensure_ascii=False) + "\n").encode()).hexdigest() for name, value in artifacts.items()}
    artifacts["p11_gate.json"] = {
        "version": VERSION, "created_at_utc": created_at_utc,
        "status": "P11_REFERENCE_CORPUS_BLOCKED",
        "p8_protocol_sha256": P8_PROTOCOL_SHA256, "p9_protocol_sha256": P9_PROTOCOL_SHA256,
        "p9_freeze_commit": P9_FREEZE_COMMIT, "p10_status": P10_STATUS,
        "schema_contract_sha256": contract["contract_sha256"],
        "artifact_sha256": artifact_hashes,
        "reference_corpus_ready": False, "all_61_validated": False,
        "p11_exact_schema_pass": False, "new_external_pcap_reserved": False,
        "p9_checkpoint_opened": False, "p9_model_imported": False,
        "p9_model_evaluated": False, "anomaly_scores_computed": False,
        "auroc_computed": False, "threshold_changed": False, "vocabulary_changed": False,
        "scaler_changed": False, "readiness": "P9_EXTERNAL_EVALUATION_NOT_AUTHORIZED",
    }
    return artifacts


def validate_artifacts(root: Path, artifacts: dict[str, dict]) -> None:
    verify_upstream(root)
    contract = artifacts["schema_contract.json"]
    if len(contract["fields"]) != 61 or contract["continuous_count"] != 58 or contract["categorical_count"] != 3:
        raise ValueError("P11 schema cardinality mismatch")
    names = [field["name"] for field in contract["fields"]]
    if len(names) != len(set(names)):
        raise ValueError("P11 duplicate feature contract")
    required_keys = {"name", "type", "units", "directionality", "aggregation_window",
                     "flow_semantics", "source_dependency", "formula", "missing_value_behavior",
                     "categorical_semantics", "reference_provenance"}
    if any(not required_keys <= field.keys() for field in contract["fields"]):
        raise ValueError("P11 incomplete field contract")
    matrix = artifacts["feature_validation_matrix.json"]
    if len(matrix["rows"]) != 61 or matrix["all_61_pass"] is not False:
        raise ValueError("P11 validation matrix gate mismatch")
    gate = artifacts["p11_gate.json"]
    calculated = {name: hashlib.sha256((json.dumps(value, indent=2, sort_keys=True,
        ensure_ascii=False) + "\n").encode()).hexdigest()
                  for name, value in artifacts.items() if name != "p11_gate.json"}
    if gate["artifact_sha256"] != calculated:
        raise ValueError("P11 artifact binding mismatch")
    if (gate["status"] != "P11_REFERENCE_CORPUS_BLOCKED" or gate["p9_model_evaluated"] is not False or
            gate["anomaly_scores_computed"] is not False or gate["new_external_pcap_reserved"] is not False):
        raise ValueError("P11 no-model/reference gate mismatch")
