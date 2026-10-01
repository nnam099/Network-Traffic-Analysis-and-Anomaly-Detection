# -*- coding: utf-8 -*-
"""
Optional LLM support for SOC alert triage.

Provider SDKs and API keys are only required when an LLM response is requested.
Importing this module is intentionally side-effect free.
"""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv


load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "none").strip().lower()
_LLM_CALL = None
_LLM_INIT_ERROR = None


def _build_client():
    if LLM_PROVIDER in {"", "none", "off", "disabled"}:
        raise ValueError("LLM_PROVIDER is disabled")

    if LLM_PROVIDER == "groq":
        from groq import Groq

        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise ValueError("Missing GROQ_API_KEY in .env")
        client = Groq(api_key=api_key)

        def call(prompt: str) -> str:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.3,
            )
            return response.choices[0].message.content

        return call

    if LLM_PROVIDER == "gemini":
        import google.generativeai as genai

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY in .env")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")

        def call(prompt: str) -> str:
            return model.generate_content(prompt).text

        return call

    if LLM_PROVIDER == "openai":
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY in .env")
        client = OpenAI(api_key=api_key)

        def call(prompt: str) -> str:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
            )
            return response.choices[0].message.content

        return call

    if LLM_PROVIDER == "anthropic":
        import anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise ValueError("Missing ANTHROPIC_API_KEY in .env")
        client = anthropic.Anthropic(api_key=api_key)

        def call(prompt: str) -> str:
            response = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        return call

    raise ValueError(f"Invalid LLM_PROVIDER: {LLM_PROVIDER}")


def get_llm_status() -> dict:
    return {
        "provider": LLM_PROVIDER,
        "initialized": _LLM_CALL is not None,
        "error": _LLM_INIT_ERROR,
    }


def _call_llm(prompt: str) -> str:
    global _LLM_CALL, _LLM_INIT_ERROR
    if _LLM_CALL is None and _LLM_INIT_ERROR is None:
        try:
            _LLM_CALL = _build_client()
        except Exception as exc:
            _LLM_INIT_ERROR = str(exc)

    if _LLM_CALL is not None:
        return _LLM_CALL(prompt)

    init_error = _LLM_INIT_ERROR or "LLM client is not initialized"
    return json.dumps(
        {
            "severity": "HIGH",
            "verdict": f"LLM is not ready: {init_error}",
            "attack_summary": "Check API key, provider selection and provider SDK installation.",
            "recommended_actions": ["Check .env", "Check API key", "Install provider SDK if needed"],
            "false_positive_risk": "UNKNOWN",
            "false_positive_reason": "N/A",
            "ood_confidence": "UNKNOWN",
            "analyst_note": init_error,
        },
        ensure_ascii=False,
    )


class SOCTriageAgent:
    SYSTEM_CONTEXT = (
        "You are a SOC analyst assistant. Analyze IDS alerts clearly and return "
        "concise Vietnamese output when asked by the dashboard."
    )

    def triage_alert(self, alert_data: dict) -> dict:
        vae_score = alert_data.get("vae_recon_score", alert_data.get("ae_score", 0))
        ood_ens_score = alert_data.get("ood_ensemble_score", "N/A")
        knn_score = alert_data.get("knn_dist_score", "N/A")
        attn_summary = alert_data.get("attention_summary", "Chua co")

        ood_ens_str = f"{ood_ens_score:.4f}" if isinstance(ood_ens_score, float) else str(ood_ens_score)
        knn_str = f"{knn_score:.4f}" if isinstance(knn_score, float) else str(knn_score)
        ood_candidate_str = "CO" if alert_data.get("is_zeroday") else "KHONG"

        prompt = f"""NHIEM VU: Phan tich alert bao mat va chi tra ve mot JSON object hop le.

=== DU LIEU ALERT ===
Alert ID        : {alert_data.get('alert_id', 'N/A')}
Thoi gian       : {alert_data.get('timestamp', 'N/A')}
Hybrid Score    : {alert_data.get('hybrid_score', 0):.4f}
VAE Recon Error : {vae_score:.4f}
OOD Ensemble    : {ood_ens_str}
KNN Distance    : {knn_str}
Phan loai       : {alert_data.get('predicted_class', 'Unknown')} (do tin cay: {alert_data.get('max_prob', 0):.1%})
OOD Candidate   : {ood_candidate_str}
SHAP            : {alert_data.get('shap_summary', 'Chua co')[:300]}
Attention Gate  : {attn_summary[:200]}
MITRE           : {alert_data.get('mitre_summary', 'Chua co')[:200]}

=== OUTPUT JSON ===
{{"severity":"CRITICAL|HIGH|MEDIUM|LOW","verdict":"<1 cau mo ta alert>","attack_summary":"<giai thich dua tren SHAP/Attention va MITRE>","recommended_actions":["<hanh dong 1>","<hanh dong 2>","<hanh dong 3>"],"false_positive_risk":"HIGH|MEDIUM|LOW","false_positive_reason":"<ly do>","ood_confidence":"HIGH|MEDIUM|LOW","analyst_note":"<ghi chu>"}}"""

        try:
            raw = _call_llm(prompt)
        except Exception as exc:
            return self._fallback_result(alert_data, vae_score, f"LLM runtime error: {exc}")

        cleaned = self._extract_json(raw)
        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError:
            result = {
                "severity": "HIGH",
                "verdict": "Phat hien traffic bat thuong - can review",
                "attack_summary": raw[:400] if raw else "Khong co du lieu",
                "recommended_actions": ["Review thu cong", "Escalate len Tier 2"],
                "false_positive_risk": "MEDIUM",
                "false_positive_reason": "LLM response parsing loi",
                "ood_confidence": "MEDIUM",
                "analyst_note": "Kiem tra lai prompt va LLM response",
            }

        result.update(self._metadata(alert_data, vae_score))
        return result

    def explain_to_analyst(self, question: str, alert_context: dict) -> str:
        safe_context = {}
        for key, value in alert_context.items():
            if key in {"shap_values", "probs"}:
                continue
            try:
                json.dumps(value)
                safe_context[key] = value
            except (TypeError, ValueError):
                safe_context[key] = str(value)

        prompt = f"""
{self.SYSTEM_CONTEXT}

Context alert hien tai:
{json.dumps(safe_context, ensure_ascii=False, indent=2, default=str)}

Cau hoi cua analyst: {question}

Tra loi ngan gon, ro rang bang Tieng Viet.
"""
        try:
            return _call_llm(prompt)
        except Exception as exc:
            return f"LLM khong phan hoi. Kiem tra API key/provider. Loi: {exc}"

    @staticmethod
    def _extract_json(raw: str) -> str:
        cleaned = (raw or "").strip()
        if "```" in cleaned:
            for part in cleaned.split("```"):
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    cleaned = part
                    break
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            return cleaned[start:end]
        return cleaned

    @staticmethod
    def _metadata(alert_data: dict, vae_score) -> dict:
        return {
            "alert_id": alert_data.get("alert_id", ""),
            "timestamp": alert_data.get("timestamp", ""),
            "hybrid_score": alert_data.get("hybrid_score", 0),
            "vae_recon_score": vae_score,
            "ood_ensemble_score": alert_data.get("ood_ensemble_score", None),
            "knn_dist_score": alert_data.get("knn_dist_score", None),
        }

    def _fallback_result(self, alert_data: dict, vae_score, note: str) -> dict:
        result = {
            "severity": "HIGH",
            "verdict": "LLM gap loi - can review thu cong",
            "attack_summary": "LLM khong phan hoi. Kiem tra API key va provider.",
            "recommended_actions": ["Review thu cong", "Kiem tra API key"],
            "false_positive_risk": "UNKNOWN",
            "false_positive_reason": "LLM error",
            "ood_confidence": "UNKNOWN",
            "analyst_note": note,
        }
        result.update(self._metadata(alert_data, vae_score))
        return result
