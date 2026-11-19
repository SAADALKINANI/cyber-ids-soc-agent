import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ids.agent import Alert, LLMPlanner, RulePlanner, TriageContext, triage, validate_report  # noqa: E402
from ids.data import CATEGORICAL, COLUMNS, MUTABLE, NUMERIC, Preprocessor  # noqa: E402
from ids.models import FTTransformer  # noqa: E402
from ids.robustness import MahalanobisDetector, make_pgd, psi  # noqa: E402


def fake_df(n=64, rng=np.random.default_rng(0)):
    d = {c: rng.integers(0, 100, n) for c in COLUMNS[:-2]}
    d["protocol_type"] = rng.choice(["tcp", "udp", "icmp"], n)
    d["service"] = rng.choice(["http", "ftp", "smtp"], n)
    d["flag"] = rng.choice(["SF", "S0"], n)
    return pd.DataFrame(d)


def tiny_model():
    pre = Preprocessor.fit(fake_df())
    return pre, FTTransformer(len(NUMERIC), pre.cardinalities, d=16, layers=1, heads=2)


def test_preprocessor_handles_unseen_categories():
    pre = Preprocessor.fit(fake_df())
    df = fake_df(8); df.loc[0, "service"] = "never_seen"
    xn, xc = pre.transform(df)
    assert xn.shape == (8, len(NUMERIC)) and xc.shape == (8, len(CATEGORICAL)) and xc[0, 1] == 0


def test_ft_transformer_forward_and_mfm():
    pre, m = tiny_model()
    xn, xc = map(torch.from_numpy, pre.transform(fake_df(16)))
    assert m(xn, xc).shape == (16, 5)
    loss = m.mfm_loss(xn, xc)
    loss.backward()
    assert torch.isfinite(loss)


def test_mahalanobis_flags_far_points():
    rng = np.random.default_rng(0)
    z = np.concatenate([rng.normal(0, 1, (300, 4)), rng.normal(5, 1, (300, 4))]); y = np.repeat([0, 1], 300)
    det = MahalanobisDetector().fit(z, y)
    assert det.score(np.full((1, 4), 30.0))[0] > np.quantile(det.score(z), 0.999)


def test_psi_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert psi(a, rng.normal(0, 1, 5000)) < 0.02 and psi(a, rng.normal(2, 1, 5000)) > 0.5


def test_pgd_only_touches_mutable_features_within_eps():
    pre, m = tiny_model()
    xn, xc = map(torch.from_numpy, pre.transform(fake_df(16)))
    y = torch.ones(16, dtype=torch.long)
    idx = [NUMERIC.index(c) for c in MUTABLE]
    adv = make_pgd(idx, eps=0.5, steps=3, allow_decrease=True)(m, xn, xc, y)
    delta = (adv - xn).abs()
    frozen = [i for i in range(len(NUMERIC)) if i not in idx]
    assert delta[:, frozen].max() == 0 and delta.max() <= 0.5 + 1e-6


def _ctx_and_alert():
    pre, m = tiny_model()
    xn, xc = pre.transform(fake_df(40))
    with torch.no_grad():
        z = m.embed(torch.from_numpy(xn), torch.from_numpy(xc)).numpy()
    det = MahalanobisDetector().fit(z, np.arange(40) % 2)
    sc = det.score(z)
    alerts = [Alert(f"A{i}", float(i), "203.0.113.5", "10.0.0.1", "http", xn[i], xc[i], raw={c: 1 for c in NUMERIC})
              for i in range(5)]
    return TriageContext(m, det, float(np.quantile(sc, 0.99)), sc, alerts), alerts[0]


def test_rule_planner_produces_valid_report_and_never_executes():
    ctx, alert = _ctx_and_alert()
    out = triage(alert, ctx, RulePlanner())
    assert out["validation_errors"] == [] or out["report"]["severity"] == "low"
    assert all(a["status"] == "PENDING_HUMAN_APPROVAL" for a in out["pending_actions"])
    assert [t["tool"] for t in out["trace"]][:1] == ["classify"]


def test_guardrail_rejects_hallucinating_llm_and_falls_back():
    ctx, alert = _ctx_and_alert()
    script = iter([json.dumps({"tool": "classify", "args": {}}),
                   json.dumps({"tool": "rm_rf", "args": {}}),                    # not on the allow-list
                   json.dumps({"final": {"summary": "Blocked the attacker.", "severity": "critical",
                                         "techniques": ["T9999"], "rationale": "x", "actions": []}})])
    planner = LLMPlanner(complete=lambda system, msgs: next(script))
    out = triage(alert, ctx, planner)
    assert out["used_fallback"] is True
    assert any("hallucinated" in e for e in out["validation_errors"])
    assert any(t["result"].get("error") == "tool not allowed" for t in out["trace"])
    assert "T9999" not in out["report"]["techniques"]


def test_validate_report_flags_executed_actions():
    errs = validate_report({"summary": "We quarantined the host", "severity": "high", "techniques": [],
                            "rationale": "", "actions": []}, [])
    assert any("executed" in e for e in errs)
