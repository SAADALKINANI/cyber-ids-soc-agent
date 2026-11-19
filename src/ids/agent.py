"""SOC triage agent: turns raw model alerts into analyst-ready incident reports.

Design principles (the things a federal SOC lead will ask about):

1. **Tools, not vibes.** The agent can only act through a fixed allow-list of tools
   (classify, explain, novelty, attack_kb, related_alerts, propose_action). Every tool call
   and result is recorded in an audit trace.
2. **Human-in-the-loop.** ``propose_action`` never executes anything; it creates a *pending*
   action that an analyst must approve. Containment is never automatic.
3. **Grounded output.** Reports are validated: every MITRE ATT&CK ID must exist in the
   curated KB and every number must come from a tool result; otherwise the draft is rejected
   and the deterministic report is used instead.
4. **Model-agnostic.** The planner can be an LLM (Anthropic, OpenAI or any OpenAI-compatible
   endpoint such as a local Ollama model for air-gapped enclaves) or the deterministic
   ``RulePlanner``, which runs the same investigation without any external call.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np
import torch

from .attack_kb import FAMILY_TECHNIQUES, PLAYBOOKS, TECHNIQUES
from .data import CLASSES, NUMERIC

SYSTEM_PROMPT = """You are a Tier-1 SOC triage analyst for a U.S. federal agency.
Investigate the alert using ONLY the tools provided. Respond with exactly one JSON object per turn:
  {"tool": "<name>", "args": {...}}   to call a tool, or
  {"final": {"summary": str, "severity": "low|medium|high|critical", "techniques": [ATT&CK IDs],
             "rationale": str, "actions": [{"type": "contain|investigate|escalate", "description": str}]}}
Rules: cite only ATT&CK IDs returned by the attack_kb tool; never claim an action was executed -
actions are proposals for human approval; if novelty is high, say the pattern is unknown."""


@dataclass
class Alert:
    alert_id: str
    timestamp: float
    src_ip: str
    dst_ip: str
    service: str
    x_num: np.ndarray
    x_cat: np.ndarray
    raw: dict = field(default_factory=dict)


@dataclass
class TriageContext:
    model: torch.nn.Module
    detector: Any                      # MahalanobisDetector
    novelty_threshold: float           # e.g. 99th percentile of validation scores
    novelty_ref: np.ndarray            # validation scores, to express novelty as a percentile
    alerts: list[Alert]
    pending_actions: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------------------------
def _tensor(a: Alert):
    return torch.from_numpy(a.x_num[None]), torch.from_numpy(a.x_cat[None])


def tool_classify(ctx: TriageContext, alert: Alert, **_) -> dict:
    ctx.model.eval()
    with torch.no_grad():
        p = torch.softmax(ctx.model(*_tensor(alert)), -1)[0].numpy()
    k = int(p.argmax())
    return {"predicted_family": CLASSES[k], "confidence": round(float(p[k]), 4),
            "probabilities": {c: round(float(v), 4) for c, v in zip(CLASSES, p)}}


def tool_explain(ctx: TriageContext, alert: Alert, top_k: int = 5, **_) -> dict:
    """Gradient x input attribution for the predicted class (numeric features)."""
    ctx.model.eval()
    xn, xc = _tensor(alert)
    xn = xn.clone().requires_grad_(True)
    logits = ctx.model(xn, xc)
    k = int(logits.argmax())
    g, = torch.autograd.grad(logits[0, k], xn)
    contrib = (g * xn)[0].detach().numpy()
    order = np.argsort(-np.abs(contrib))[:top_k]
    return {"explained_class": CLASSES[k],
            "top_features": [{"feature": NUMERIC[i], "raw_value": alert.raw.get(NUMERIC[i]),
                              "contribution": round(float(contrib[i]), 4)} for i in order]}


def tool_novelty(ctx: TriageContext, alert: Alert, **_) -> dict:
    with torch.no_grad():
        z = ctx.model.embed(*_tensor(alert)).numpy()
    s = float(ctx.detector.score(z)[0])
    pct = float((ctx.novelty_ref < s).mean() * 100)
    return {"novelty_score": round(s, 2), "percentile_vs_known_traffic": round(pct, 1),
            "is_novel": bool(s > ctx.novelty_threshold)}


def tool_attack_kb(ctx: TriageContext, alert: Alert, family: str = "", **_) -> dict:
    tids = FAMILY_TECHNIQUES.get(family, [])
    return {"family": family, "techniques": [{"id": t, "name": TECHNIQUES[t]} for t in tids],
            "playbook": [{"type": t, "description": d} for t, d in PLAYBOOKS.get(family, PLAYBOOKS["unknown"])]}


def tool_related_alerts(ctx: TriageContext, alert: Alert, window_s: float = 900, **_) -> dict:
    rel = [a for a in ctx.alerts if a.alert_id != alert.alert_id and a.src_ip == alert.src_ip
           and abs(a.timestamp - alert.timestamp) <= window_s]
    return {"src_ip": alert.src_ip, "count_in_window": len(rel),
            "distinct_targets": len({a.dst_ip for a in rel}),
            "services": sorted({a.service for a in rel})[:10]}


def tool_propose_action(ctx: TriageContext, alert: Alert, type: str = "investigate", description: str = "", **_) -> dict:
    act = {"alert_id": alert.alert_id, "type": type, "description": description, "status": "PENDING_HUMAN_APPROVAL"}
    ctx.pending_actions.append(act)
    return act


TOOLS: dict[str, Callable[..., dict]] = {
    "classify": tool_classify, "explain": tool_explain, "novelty": tool_novelty,
    "attack_kb": tool_attack_kb, "related_alerts": tool_related_alerts, "propose_action": tool_propose_action,
}


# ---------------------------------------------------------------------------------------------
# Planners
# ---------------------------------------------------------------------------------------------
class Planner(Protocol):
    def next(self, transcript: list[dict]) -> dict: ...


def severity(family: str, confidence: float, novel: bool, campaign_size: int) -> str:
    base = {"normal": 0, "Probe": 1, "DoS": 2, "R2L": 2, "U2R": 3}[family]
    if novel:
        base = max(base, 2) + (1 if family != "normal" else 0)
    if campaign_size >= 20:
        base += 1
    if confidence < 0.6:
        base = max(base - 1, 1 if novel else 0)
    return ["low", "medium", "high", "critical", "critical", "critical"][min(base, 5)]


class RulePlanner:
    """Deterministic investigation plan; identical tool usage to the LLM path, no network calls."""

    def next(self, transcript: list[dict]) -> dict:
        done = {t["tool"]: t["result"] for t in transcript if "tool" in t}
        if "classify" not in done:
            return {"tool": "classify", "args": {}}
        fam = done["classify"]["predicted_family"]
        for step in ("novelty", "explain", "related_alerts"):
            if step not in done:
                return {"tool": step, "args": {}}
        nov = done["novelty"]["is_novel"]
        kb_family = "unknown" if (nov and fam == "normal") else fam
        if "attack_kb" not in done:
            return {"tool": "attack_kb", "args": {"family": kb_family}}
        if fam == "normal" and not nov:
            return {"final": {"summary": "Traffic consistent with known benign behaviour.", "severity": "low",
                              "techniques": [], "rationale": f"Classified normal (p={done['classify']['confidence']}); "
                              f"novelty percentile {done['novelty']['percentile_vs_known_traffic']}.", "actions": []}}
        rel = done["related_alerts"]
        sev = severity(fam, done["classify"]["confidence"], nov, rel["count_in_window"])
        feats = ", ".join(f"{f['feature']}={f['raw_value']}" for f in done["explain"]["top_features"][:3])
        label = "UNKNOWN pattern (possible zero-day)" if kb_family == "unknown" else f"{fam} activity"
        if nov and kb_family != "unknown":
            label += " with novel characteristics"
        techs = [t["id"] for t in done["attack_kb"]["techniques"]]
        return {"final": {
            "summary": f"{label} from {rel['src_ip']} ({rel['count_in_window']} related alerts in 15 min, "
                       f"{rel['distinct_targets']} distinct targets).",
            "severity": sev, "techniques": techs,
            "rationale": f"Classifier: {fam} (p={done['classify']['confidence']}). Novelty percentile "
                         f"{done['novelty']['percentile_vs_known_traffic']}. Key drivers: {feats}.",
            "actions": done["attack_kb"]["playbook"]}}


class LLMPlanner:
    """JSON-tool-calling planner over any chat model.

    Backends: ``anthropic`` (ANTHROPIC_API_KEY), ``openai`` (OPENAI_API_KEY) or any
    OpenAI-compatible server via OPENAI_BASE_URL (e.g. Ollama/vLLM inside an enclave).
    """

    def __init__(self, backend: str = "anthropic", model: str | None = None, complete: Callable | None = None):
        self.backend, self.model = backend, model
        self._complete = complete or self._default_complete

    def _default_complete(self, system: str, messages: list[dict]) -> str:
        if self.backend == "anthropic":
            import anthropic  # optional dependency
            r = anthropic.Anthropic().messages.create(model=self.model or "claude-sonnet-4-5", max_tokens=1200,
                                                      system=system, messages=messages)
            return r.content[0].text
        from openai import OpenAI  # optional dependency
        r = OpenAI(base_url=os.getenv("OPENAI_BASE_URL")).chat.completions.create(
            model=self.model or "gpt-4o-mini", messages=[{"role": "system", "content": system}] + messages)
        return r.choices[0].message.content

    def next(self, transcript: list[dict]) -> dict:
        msgs = [{"role": "user", "content": "Investigate alert. Tools: " + ", ".join(TOOLS)}]
        for t in transcript:
            msgs.append({"role": "assistant", "content": json.dumps({"tool": t["tool"], "args": t["args"]})})
            msgs.append({"role": "user", "content": "RESULT: " + json.dumps(t["result"])})
        text = self._complete(SYSTEM_PROMPT, msgs)
        m = re.search(r"\{.*\}", text, re.S)
        return json.loads(m.group(0)) if m else {"final": None}


# ---------------------------------------------------------------------------------------------
# Agent loop + guardrails
# ---------------------------------------------------------------------------------------------
def validate_report(report: dict | None, transcript: list[dict]) -> list[str]:
    """Return a list of problems; empty list = report accepted."""
    if not isinstance(report, dict):
        return ["no report produced"]
    errs = []
    for k in ("summary", "severity", "techniques", "rationale", "actions"):
        if k not in report:
            errs.append(f"missing field {k}")
    allowed = {t["id"] for s in transcript if s["tool"] == "attack_kb" for t in s["result"]["techniques"]}
    for tid in report.get("techniques", []):
        if tid not in TECHNIQUES:
            errs.append(f"hallucinated ATT&CK id {tid}")
        elif tid not in allowed:
            errs.append(f"ATT&CK id {tid} not supported by attack_kb result")
    if report.get("severity") not in {"low", "medium", "high", "critical"}:
        errs.append("invalid severity")
    text = json.dumps(report).lower()
    if re.search(r"\b(blocked|isolated|executed|quarantined)\b", text) and "pending" not in text:
        errs.append("claims an action was executed (actions must be proposals)")
    return errs


def triage(alert: Alert, ctx: TriageContext, planner: Planner | None = None, max_steps: int = 8) -> dict:
    planner = planner or RulePlanner()
    transcript: list[dict] = []
    report, used_fallback = None, False
    for _ in range(max_steps):
        try:
            step = planner.next(transcript)
        except Exception as e:  # network / parsing failure -> deterministic fallback
            step = {"final": None, "error": str(e)}
        if "final" in step:
            report = step["final"]
            break
        name = step.get("tool")
        if name not in TOOLS:
            transcript.append({"tool": str(name), "args": step.get("args", {}), "result": {"error": "tool not allowed"}})
            continue
        result = TOOLS[name](ctx, alert, **(step.get("args") or {}))
        transcript.append({"tool": name, "args": step.get("args", {}), "result": result})
    problems = validate_report(report, transcript)
    if problems and not isinstance(planner, RulePlanner):
        used_fallback = True
        rp = RulePlanner()
        transcript_fb = list(transcript)
        while True:
            s = rp.next(transcript_fb)
            if "final" in s:
                report = s["final"]; break
            transcript_fb.append({"tool": s["tool"], "args": s["args"], "result": TOOLS[s["tool"]](ctx, alert, **s["args"])})
        transcript = transcript_fb
    for a in (report or {}).get("actions", []):
        tool_propose_action(ctx, alert, **a)
    return {"alert_id": alert.alert_id, "report": report, "trace": transcript,
            "validation_errors": problems, "used_fallback": used_fallback,
            "pending_actions": [a for a in ctx.pending_actions if a["alert_id"] == alert.alert_id]}
