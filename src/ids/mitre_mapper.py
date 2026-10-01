from __future__ import annotations

from typing import Iterable, Sequence


CLASS_NAMES = ["Normal", "DoS", "Exploits", "Reconnaissance", "Generic"]


TECHNIQUE_LIBRARY = {
    "T1498": {
        "name": "Network Denial of Service",
        "tactic": "Impact",
        "response": ["Rate-limit or block source ranges", "Check upstream firewall and WAF counters"],
    },
    "T1499": {
        "name": "Endpoint Denial of Service",
        "tactic": "Impact",
        "response": ["Inspect target service saturation", "Correlate host CPU, memory and socket exhaustion"],
    },
    "T1190": {
        "name": "Exploit Public-Facing Application",
        "tactic": "Initial Access",
        "response": ["Review exposed service logs", "Check recent CVEs and patch level for destination service"],
    },
    "T1203": {
        "name": "Exploitation for Client Execution",
        "tactic": "Execution",
        "response": ["Inspect payload-bearing sessions", "Hunt for follow-up process or script execution"],
    },
    "T1068": {
        "name": "Exploitation for Privilege Escalation",
        "tactic": "Privilege Escalation",
        "response": ["Review endpoint privilege changes", "Search for abnormal child processes"],
    },
    "T1595": {
        "name": "Active Scanning",
        "tactic": "Reconnaissance",
        "response": ["Correlate with port/service scan telemetry", "Throttle or block repeat scanners"],
    },
    "T1046": {
        "name": "Network Service Discovery",
        "tactic": "Discovery",
        "response": ["Check connection fan-out and destination diversity", "Review asset inventory exposure"],
    },
    "T1071": {
        "name": "Application Layer Protocol",
        "tactic": "Command and Control",
        "response": ["Inspect protocol legitimacy", "Hunt for repeated low-volume beaconing"],
    },
    "T1543": {
        "name": "Create or Modify System Process",
        "tactic": "Persistence",
        "response": ["Inspect service creation events", "Check startup persistence locations"],
    },
    "T1078": {
        "name": "Valid Accounts",
        "tactic": "Defense Evasion",
        "response": ["Review authentication logs", "Check impossible travel and new device signals"],
    },
    "T1091": {
        "name": "Replication Through Removable Media",
        "tactic": "Lateral Movement",
        "response": ["Review lateral movement telemetry", "Check removable media and autorun events"],
    },
    "T1210": {
        "name": "Exploitation of Remote Services",
        "tactic": "Lateral Movement",
        "response": ["Inspect remote service authentication", "Patch vulnerable internal services"],
    },
    "T1055": {
        "name": "Process Injection",
        "tactic": "Defense Evasion",
        "response": ["Collect endpoint process tree", "Check memory injection detections"],
    },
    "T1059": {
        "name": "Command and Scripting Interpreter",
        "tactic": "Execution",
        "response": ["Review shell/script telemetry", "Contain host if suspicious execution is confirmed"],
    },
    "T1110": {
        "name": "Brute Force",
        "tactic": "Credential Access",
        "response": ["Check failed login velocity", "Temporarily enforce account lockout or MFA challenge"],
    },
    "T1040": {
        "name": "Network Sniffing",
        "tactic": "Credential Access",
        "response": ["Review promiscuous-mode or SPAN misuse", "Check credential exposure on cleartext protocols"],
    },
    "T1030": {
        "name": "Data Transfer Size Limits",
        "tactic": "Exfiltration",
        "response": ["Inspect byte volume and destination reputation", "Apply egress controls for suspicious flows"],
    },
    "T1132": {
        "name": "Data Encoding",
        "tactic": "Command and Control",
        "response": ["Inspect encoded payload patterns", "Correlate with DNS/HTTP beaconing"],
    },
}


CLASS_TO_TECHNIQUES = {
    "DoS": [
        ("T1498", 0.86, "Traffic pattern resembles volumetric or protocol-level denial of service."),
        ("T1499", 0.70, "Classifier indicates service availability impact risk."),
    ],
    "Exploits": [
        ("T1190", 0.82, "Exploit class maps to public service exploitation in network telemetry."),
        ("T1203", 0.68, "Payload-like traffic can indicate client or application exploitation."),
        ("T1068", 0.55, "Follow-up privilege escalation should be checked if host telemetry confirms compromise."),
    ],
    "Reconnaissance": [
        ("T1595", 0.88, "Reconnaissance traffic commonly reflects active probing or scanning."),
        ("T1046", 0.80, "Connection count and service diversity may indicate service discovery."),
    ],
    "Generic": [
        ("T1071", 0.62, "Generic attack traffic may use common application protocols for C2 or abuse."),
        ("T1190", 0.50, "Generic abnormal service traffic can still represent exploitation attempts."),
    ],
    "Backdoors": [
        ("T1543", 0.64, "Backdoor activity may lead to service or process persistence."),
        ("T1078", 0.58, "Account misuse should be checked when persistence indicators are weak."),
    ],
    "Worms": [
        ("T1210", 0.80, "Worm behavior often spreads through vulnerable remote services."),
        ("T1091", 0.50, "Replication vectors should be considered if endpoint evidence supports it."),
    ],
    "Shellcode": [
        ("T1055", 0.76, "Shellcode-like behavior can indicate injected execution."),
        ("T1059", 0.64, "Command execution should be checked after suspected payload delivery."),
    ],
    "Fuzzers": [
        ("T1595", 0.72, "Fuzzing traffic frequently appears as probing or active scanning."),
        ("T1110", 0.42, "Credential attack correlation is needed before treating this as brute force."),
    ],
    "Analysis": [
        ("T1040", 0.58, "Analysis traffic may involve capture or inspection of network data."),
        ("T1046", 0.54, "Service discovery is a plausible analyst-facing interpretation."),
    ],
    "Zero-Day": [
        ("T1190", 0.48, "Unknown attack is safest to triage first as possible exposed-service exploitation."),
    ],
}


FEATURE_RULES = [
    (
        {"sbytes", "dbytes", "sload", "dload", "log_total_bytes", "log_sbytes", "log_dbytes"},
        [("T1030", 0.72, "Byte and load features are dominant, suggesting abnormal transfer volume."),
         ("T1498", 0.66, "High traffic asymmetry or volume can represent network DoS pressure.")],
    ),
    (
        {"ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_srv_src", "ct_dst_src_ltm", "ct_dst_sport_ltm"},
        [("T1046", 0.76, "Connection-count features are dominant, suggesting service discovery or scanning."),
         ("T1595", 0.70, "Repeated destination/service probing is consistent with active scanning.")],
    ),
    (
        {"dur", "sintpkt", "dintpkt", "intpkt_ratio", "intpkt_diff"},
        [("T1132", 0.52, "Timing and duration anomalies can indicate encoded or staged C2 traffic."),
         ("T1071", 0.50, "Application protocol abuse should be checked for repeated timing patterns.")],
    ),
    (
        {"tcprtt", "synack", "ackdat", "handshake_ratio", "incomplete_tcp", "state_num"},
        [("T1190", 0.58, "Handshake and TCP-state anomalies can accompany service exploitation attempts."),
         ("T1498", 0.55, "Incomplete TCP patterns may also indicate protocol-level DoS behavior.")],
    ),
]


def _feature_names(top_features: Iterable) -> set[str]:
    names = set()
    for item in top_features or []:
        if isinstance(item, (list, tuple)) and item:
            names.add(str(item[0]))
        elif isinstance(item, dict) and item.get("feature"):
            names.add(str(item["feature"]))
    return names


def _technique(tech_id: str, confidence: float, rationale: str, evidence=None) -> dict:
    base = TECHNIQUE_LIBRARY[tech_id]
    return {
        "id": tech_id,
        "name": base["name"],
        "tactic": base["tactic"],
        "confidence": round(float(confidence), 2),
        "rationale": rationale,
        "evidence": list(evidence) if evidence else [],
        "response_actions": base["response"],
        "url": f"https://attack.mitre.org/techniques/{tech_id}/",
    }


def _dedupe_rank(techniques: list[dict]) -> list[dict]:
    best: dict[str, dict] = {}
    for technique in techniques:
        existing = best.get(technique["id"])
        if existing is None or technique["confidence"] > existing["confidence"]:
            best[technique["id"]] = technique
        elif existing:
            existing["evidence"] = sorted(set(existing.get("evidence", []) + technique.get("evidence", [])))
    return sorted(best.values(), key=lambda t: t["confidence"], reverse=True)


class MITREMapper:
    def __init__(self):
        self.mapping = CLASS_TO_TECHNIQUES

    def map_known_attack(self, class_idx: int, class_names: Sequence[str] | None = None,
                         top_features=None, max_items: int = 4) -> dict | None:
        class_names = list(class_names or CLASS_NAMES)
        if class_idx < 0 or class_idx >= len(class_names):
            class_name = "Generic"
        else:
            class_name = class_names[class_idx]
        if class_name == "Normal":
            return None

        techniques = [
            _technique(tech_id, confidence, rationale, evidence=[f"classifier:{class_name}"])
            for tech_id, confidence, rationale in self.mapping.get(class_name, self.mapping["Generic"])
        ]
        techniques.extend(self._map_features(top_features))
        techniques = _dedupe_rank(techniques)[:max_items]

        return {
            "attack_class": class_name,
            "mapping_mode": "known_attack",
            "techniques": techniques,
            "primary_tactic": techniques[0]["tactic"] if techniques else None,
            "confidence": self._overall_confidence(techniques),
            "coverage_note": "MITRE mapping is heuristic and should be validated with endpoint, firewall and SIEM evidence.",
        }

    def map_zeroday(self, ae_score: float, top_shap_features=None, max_items: int = 5) -> dict:
        feature_based = self._map_features(top_shap_features)
        if not feature_based:
            feature_based = [
                _technique(tech_id, min(confidence + ae_score * 0.15, 0.75), rationale,
                           evidence=[f"ae_score:{ae_score:.3f}"])
                for tech_id, confidence, rationale in self.mapping["Zero-Day"]
            ]
        else:
            for technique in feature_based:
                technique["confidence"] = round(min(technique["confidence"] + ae_score * 0.10, 0.92), 2)
                technique["evidence"].append(f"ae_score:{ae_score:.3f}")

        techniques = _dedupe_rank(feature_based)[:max_items]
        return {
            "attack_class": "Zero-Day",
            "mapping_mode": "zero_day_hypothesis",
            "ae_reconstruction_error": round(float(ae_score), 4),
            "techniques": techniques,
            "suspected_techniques": techniques,
            "primary_tactic": techniques[0]["tactic"] if techniques else None,
            "confidence": self._overall_confidence(techniques),
            "coverage_note": "Zero-day mapping is a hypothesis based on model anomaly score and dominant features.",
        }

    def _map_features(self, top_features) -> list[dict]:
        names = _feature_names(top_features)
        techniques: list[dict] = []
        if not names:
            return techniques
        for feature_set, candidates in FEATURE_RULES:
            hits = sorted(names & feature_set)
            if not hits:
                continue
            for tech_id, confidence, rationale in candidates:
                techniques.append(_technique(tech_id, confidence, rationale, evidence=hits))
        return techniques

    @staticmethod
    def _overall_confidence(techniques: list[dict]) -> str:
        if not techniques:
            return "LOW"
        score = max(t.get("confidence", 0) for t in techniques)
        if score >= 0.75:
            return "HIGH"
        if score >= 0.55:
            return "MEDIUM"
        return "LOW"

    def format_for_llm(self, mitre_result) -> str:
        if not mitre_result:
            return "Traffic appears normal; no MITRE ATT&CK mapping is attached."

        lines = [
            f"Attack class: {mitre_result.get('attack_class', 'Unknown')}",
            f"Mapping mode: {mitre_result.get('mapping_mode', 'unknown')}",
            f"Overall confidence: {mitre_result.get('confidence', 'LOW')}",
        ]
        techniques = mitre_result.get("techniques") or mitre_result.get("suspected_techniques", [])
        if techniques:
            lines.append("Mapped MITRE ATT&CK techniques:")
            for technique in techniques:
                lines.append(
                    f"- [{technique['id']}] {technique['name']} | "
                    f"Tactic={technique['tactic']} | Confidence={technique['confidence']} | "
                    f"Why={technique.get('rationale', '')}"
                )
        if mitre_result.get("coverage_note"):
            lines.append(f"Note: {mitre_result['coverage_note']}")
        return "\n".join(lines)
