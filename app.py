"""SOC Triage Console.  Run:  streamlit run app.py   (after scripts/run_pipeline.py)"""
import json
import sys
from pathlib import Path

import streamlit as st
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from ids.agent import Alert, TriageContext, triage  # noqa: E402
from ids.data import CATEGORICAL, NUMERIC, Preprocessor, load  # noqa: E402
from ids.models import FTTransformer  # noqa: E402
from ids.robustness import MahalanobisDetector  # noqa: E402

st.set_page_config(page_title="SOC Triage Console", layout="wide")


def ensure_data():
    """Fresh clone: fetch NSL-KDD (~20 MB) from the same mirror as scripts/download_data.sh."""
    raw = ROOT / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    import urllib.request
    for f in ("KDDTrain+.txt", "KDDTest+.txt"):
        if not (raw / f).exists():
            urllib.request.urlretrieve(f"https://raw.githubusercontent.com/defcom17/NSL_KDD/master/{f}", raw / f)


@st.cache_resource(show_spinner="Loading model and NSL-KDD data...")
def setup():
    ensure_data()
    ck = torch.load(ROOT / "artifacts" / "ids_model.pt", weights_only=False)
    pre = Preprocessor(**ck["pre"])
    model = FTTransformer(len(NUMERIC), ck["cards"]); model.load_state_dict(ck["state_dict"]); model.eval()
    det = MahalanobisDetector(); det.mu, det.prec, det.classes = ck["det_mu"], ck["det_prec"], ck["det_classes"]
    test = load(ROOT / "data" / "raw").test
    return pre, model, det, ck["novelty_threshold"], ck["val_scores"], test


pre, model, det, thr, ref, test = setup()
M = json.loads((ROOT / "results" / "metrics.json").read_text())
st.title("SOC Triage Console - ML detection + agentic triage")
c = st.columns(4)
dm = M["models"][M["deployed_model"]]
c[0].metric("Attack detection rate (KDDTest+)", f"{dm['detection_rate']:.1%}")
c[1].metric("False-positive rate", f"{dm['false_positive_rate']:.1%}")
c[2].metric("Zero-day detection (with novelty)", f"{M['zero_day']['novel_detected_classifier_plus_novelty']:.0%}",
            f"+{M['zero_day']['novel_detected_classifier_plus_novelty'] - M['zero_day']['novel_detected_classifier_only']:.0%}")
c[3].metric("Agent reports passing guardrails", f"{M['agent']['reports_passing_validation']:.0%}")

st.sidebar.header("Pick traffic to triage")
kind = st.sidebar.radio("Sample from", ["never-seen attack types (zero-day)", "known attacks", "benign"])
mask = {"never-seen attack types (zero-day)": test.novel_attack == 1,
        "known attacks": (test.is_attack == 1) & (test.novel_attack == 0), "benign": test.is_attack == 0}[kind]
seed = st.sidebar.number_input("sample #", 0, 10_000, 7)
row = test[mask].sample(1, random_state=int(seed))
xn, xc = pre.transform(row)
alert = Alert("ALRT-LIVE", 0.0, "203.0.113.77", "10.0.1.20", row.service.iloc[0], xn[0], xc[0],
              raw={k: row[k].iloc[0] for k in NUMERIC + CATEGORICAL})
ctx = TriageContext(model, det, thr, ref, [alert])
out = triage(alert, ctx)
r = out["report"]
left, right = st.columns([3, 2])
with left:
    st.subheader(f"Incident report · severity **{r['severity'].upper()}**")
    st.write(r["summary"]); st.caption(r["rationale"])
    st.write("**MITRE ATT&CK:** " + (", ".join(r["techniques"]) or "none"))
    st.write("**Proposed actions (require analyst approval):**")
    for a in out["pending_actions"]:
        st.checkbox(f"[{a['type']}] {a['description']}", key=a["description"])
    st.caption(f"Ground truth (hidden from the agent): `{row.attack.iloc[0]}`")
with right:
    st.subheader("Agent trace (audit log)")
    for t in out["trace"]:
        with st.expander(f"tool: {t['tool']}"):
            st.json(t["result"])
st.divider()
st.image(str(ROOT / "results" / "figures" / "model_comparison.png"))
cols = st.columns(2)
cols[0].image(str(ROOT / "results" / "figures" / "label_efficiency.png"))
cols[1].image(str(ROOT / "results" / "figures" / "adversarial.png"))
