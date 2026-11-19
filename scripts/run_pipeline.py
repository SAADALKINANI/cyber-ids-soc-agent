"""Full experiment: baselines -> FT-Transformer (+SSL) -> zero-day detection -> drift ->
adversarial robustness -> SOC agent evaluation.

    python scripts/run_pipeline.py            # ~30-40 min on a 2-core CPU
    python scripts/run_pipeline.py --quick    # smoke test (~3 min)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.metrics import (accuracy_score, classification_report, f1_score,  # noqa: E402
                             roc_auc_score)

from ids.agent import Alert, TriageContext, triage  # noqa: E402
from ids.attack_kb import ATTACK_TECHNIQUES  # noqa: E402
from ids.data import CATEGORICAL, CLASSES, MUTABLE, NUMERIC, load  # noqa: E402
from ids.models import MLP, FTTransformer  # noqa: E402
from ids.robustness import MahalanobisDetector, energy_score, make_pgd, mmd_rbf, psi  # noqa: E402
from ids.train import (balanced_weights, embed, fit_classifier, predict_logits,  # noqa: E402
                       pretrain_mfm)

FIG = ROOT / "results" / "figures"
CACHE = ROOT / "artifacts" / "cache"


def cached_fit(name, model, fit_fn):
    """Restart-safe training: reuse a finished model if its checkpoint exists."""
    ck = CACHE / f"{name}.pt"
    if ck.exists():
        c = torch.load(ck, weights_only=False)
        model.load_state_dict(c["state"]); print(f"  [{name}] loaded from cache", flush=True)
        return model, c["info"]
    model, info = fit_fn(model)
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "info": info}, ck)
    return model, info


def evaluate(y, pred, test) -> dict:
    is_att, pred_att = test["is_attack"].to_numpy(), (pred != 0).astype(int)
    novel = test["novel_attack"].to_numpy() == 1
    normal = is_att == 0
    rep = classification_report(y, pred, labels=range(5), target_names=CLASSES, output_dict=True, zero_division=0)
    return {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"),
            "detection_rate": float(pred_att[is_att == 1].mean()),
            "false_positive_rate": float(pred_att[normal].mean()),
            "binary_f1": f1_score(is_att, pred_att),
            "novel_attack_detection_rate": float(pred_att[novel].mean()),
            "per_class_recall": {c: rep[c]["recall"] for c in CLASSES}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    FIG.mkdir(parents=True, exist_ok=True); (ROOT / "artifacts").mkdir(exist_ok=True)
    E = dict(pre=1 if args.quick else 3, ft=2 if args.quick else 8, adv=1 if args.quick else 2)
    log = lambda *a: print(*a, flush=True)  # noqa: E731

    sp = load(ROOT / "data" / "raw")
    if args.quick:
        sp.train = sp.train.sample(15000, random_state=0).reset_index(drop=True)
    pre = sp.pre
    Xn, Xc = pre.transform(sp.train); y = sp.train["y"].to_numpy()
    Vn, Vc = pre.transform(sp.val); vy = sp.val["y"].to_numpy()
    Tn, Tc = pre.transform(sp.test); ty = sp.test["y"].to_numpy()
    cw = balanced_weights(y, 5)
    R: dict = {"data": {"train": len(y), "val": len(vy), "test": len(ty),
                        "test_novel_attack_records": int(sp.test["novel_attack"].sum()),
                        "test_novel_attack_types": sorted(set(sp.test.loc[sp.test.novel_attack == 1, "attack"]))},
               "models": {}}
    log(json.dumps(R["data"]))

    # ---------------- Baselines ----------------
    t0 = time.time()
    flat = lambda n, c: np.concatenate([n, c], 1)  # noqa: E731
    sw = cw[y]
    xg = xgb.XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.1, subsample=0.8,
                           colsample_bytree=0.8, tree_method="hist", n_jobs=2, early_stopping_rounds=30)
    xg.fit(flat(Xn, Xc), y, sample_weight=sw, eval_set=[(flat(Vn, Vc), vy)], verbose=False)
    R["models"]["xgboost"] = evaluate(ty, xg.predict(flat(Tn, Tc)), sp.test) | {"train_seconds": time.time() - t0}
    log("xgboost", {k: round(v, 4) for k, v in R["models"]["xgboost"].items() if isinstance(v, float)})

    mlp, info = cached_fit("mlp", MLP(len(NUMERIC), pre.cardinalities), lambda m: fit_classifier(
        m, Xn, Xc, y, Vn, Vc, vy, epochs=E["ft"], class_weight=cw, log=lambda *_: None))
    R["models"]["mlp"] = evaluate(ty, predict_logits(mlp, Tn, Tc).argmax(1), sp.test) | info
    log("mlp", round(R["models"]["mlp"]["macro_f1"], 4))

    # ---------------- FT-Transformer: scratch vs self-supervised pre-training ----------------
    ft_scratch, info = cached_fit("ft_scratch", FTTransformer(len(NUMERIC), pre.cardinalities), lambda m: fit_classifier(
        m, Xn, Xc, y, Vn, Vc, vy, epochs=E["ft"], class_weight=cw, log=log))
    R["models"]["ft_transformer"] = evaluate(ty, predict_logits(ft_scratch, Tn, Tc).argmax(1), sp.test) | info
    log("ft scratch", round(R["models"]["ft_transformer"]["macro_f1"], 4))

    log("Self-supervised pre-training (masked feature modelling, no labels)")
    torch.manual_seed(0)
    def _pre(m):
        return m, {"loss": pretrain_mfm(m, Xn, Xc, epochs=E["pre"], log=log)}
    encoder, pinfo = cached_fit("ssl_encoder", FTTransformer(len(NUMERIC), pre.cardinalities), _pre)
    R["ssl_pretrain_loss"] = pinfo["loss"]
    pre_state = {k: v.clone() for k, v in encoder.state_dict().items()}
    ft_ssl = FTTransformer(len(NUMERIC), pre.cardinalities); ft_ssl.load_state_dict(pre_state)
    ft_ssl, info = cached_fit("ft_ssl", ft_ssl, lambda m: fit_classifier(
        m, Xn, Xc, y, Vn, Vc, vy, epochs=E["ft"], lr=5e-4, class_weight=cw, log=log))
    R["models"]["ft_transformer_ssl"] = evaluate(ty, predict_logits(ft_ssl, Tn, Tc).argmax(1), sp.test) | info
    log("ft ssl", round(R["models"]["ft_transformer_ssl"]["macro_f1"], 4))

    # ---------------- Label efficiency ----------------
    R["label_efficiency"] = {}
    le_ck = CACHE / "label_efficiency.json"
    if le_ck.exists():
        R["label_efficiency"] = json.loads(le_ck.read_text())
    for frac in ([0.01] if args.quick else [0.01, 0.05]):
        if f"{int(frac * 100)}%" in R["label_efficiency"]:
            continue
        idx = sp.train.groupby("y", group_keys=False).apply(
            lambda g: g.sample(max(3, int(len(g) * frac)), random_state=1)).index.to_numpy()
        row = {}
        torch.manual_seed(1)
        m = FTTransformer(len(NUMERIC), pre.cardinalities)
        m, _ = fit_classifier(m, Xn[idx], Xc[idx], y[idx], Vn, Vc, vy, epochs=E["ft"] * 2, class_weight=cw, patience=6, log=lambda *_: None)
        row["ft_scratch"] = f1_score(ty, predict_logits(m, Tn, Tc).argmax(1), average="macro")
        m = FTTransformer(len(NUMERIC), pre.cardinalities); m.load_state_dict(pre_state)
        m, _ = fit_classifier(m, Xn[idx], Xc[idx], y[idx], Vn, Vc, vy, epochs=E["ft"] * 2, lr=5e-4, class_weight=cw, patience=6, log=lambda *_: None)
        row["ft_ssl_pretrained"] = f1_score(ty, predict_logits(m, Tn, Tc).argmax(1), average="macro")
        xg_s = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, tree_method="hist", n_jobs=2)
        xg_s.fit(flat(Xn[idx], Xc[idx]), y[idx], sample_weight=cw[y[idx]])
        row["xgboost"] = f1_score(ty, xg_s.predict(flat(Tn, Tc)), average="macro")
        row["n_labels"] = int(len(idx))
        R["label_efficiency"][f"{int(frac * 100)}%"] = row
        CACHE.mkdir(parents=True, exist_ok=True); le_ck.write_text(json.dumps(R["label_efficiency"]))
        log("label efficiency", frac, row)

    # Pick the deployed model on *validation* macro-F1
    best_name = max(["ft_transformer", "ft_transformer_ssl"], key=lambda k: R["models"][k]["best_val_macro_f1"])
    best = ft_ssl if best_name == "ft_transformer_ssl" else ft_scratch
    R["deployed_model"] = best_name

    # ---------------- Zero-day detection ----------------
    Ztr, Zv, Zt = embed(best, Xn, Xc), embed(best, Vn, Vc), embed(best, Tn, Tc)
    det = MahalanobisDetector().fit(Ztr, y)
    sv, st = det.score(Zv), det.score(Zt)
    thr = float(np.quantile(sv, 0.99))
    novel = sp.test["novel_attack"].to_numpy() == 1
    known = ~novel
    lt = predict_logits(best, Tn, Tc)
    pred = lt.argmax(1)
    cls_flag = pred != 0
    nov_flag = st > thr
    combined = cls_flag | nov_flag
    normal = sp.test["is_attack"].to_numpy() == 0
    R["zero_day"] = {
        "auroc_mahalanobis_novel_vs_known": roc_auc_score(novel, st),
        "auroc_energy_novel_vs_known": roc_auc_score(novel, energy_score(lt)),
        "auroc_mahalanobis_novel_vs_normal": roc_auc_score(novel[novel | normal], st[novel | normal]),
        "threshold_p99_val": thr,
        "novel_detected_classifier_only": float(cls_flag[novel].mean()),
        "novel_detected_classifier_plus_novelty": float(combined[novel].mean()),
        "fpr_classifier_only": float(cls_flag[normal].mean()),
        "fpr_classifier_plus_novelty": float(combined[normal].mean()),
        "per_novel_attack_recall": sp.test[novel].assign(hit=combined[novel]).groupby("attack")["hit"].mean().round(3).to_dict(),
    }
    log("zero-day", {k: v for k, v in R["zero_day"].items() if k != "per_novel_attack_recall"})

    # ---------------- Drift ----------------
    R["drift"] = {
        "mmd_train_vs_val (no drift reference)": mmd_rbf(Ztr, Zv),
        "mmd_train_vs_test": mmd_rbf(Ztr, Zt),
        "psi_top_features": dict(sorted({c: psi(Xn[:, i], Tn[:, i]) for i, c in enumerate(NUMERIC)}.items(),
                                        key=lambda kv: -kv[1])[:8]),
    }
    log("drift", R["drift"])

    # ---------------- Adversarial robustness ----------------
    mut_idx = [NUMERIC.index(c) for c in MUTABLE]
    pgd = make_pgd(mut_idx, eps=1.0, steps=10, allow_decrease=True)
    att_known = np.where((ty != 0) & known)[0]
    rng = np.random.default_rng(0)
    sub = rng.choice(att_known, min(3000, len(att_known)), replace=False)

    def adv_detection(model):
        xa = pgd(model, torch.from_numpy(Tn[sub]), torch.from_numpy(Tc[sub]), torch.from_numpy(ty[sub]))
        clean = predict_logits(model, Tn[sub], Tc[sub]).argmax(1) != 0
        with torch.no_grad():
            model.eval(); advp = model(xa, torch.from_numpy(Tc[sub])).argmax(1).numpy() != 0
        return float(clean.mean()), float(advp.mean())

    c0, a0 = adv_detection(best)
    log(f"adversarial: detection clean {c0:.3f} -> under attack {a0:.3f}")
    robust = FTTransformer(len(NUMERIC), pre.cardinalities); robust.load_state_dict(best.state_dict())
    adv_idx = rng.choice(len(y), min(20000, len(y)), replace=False)
    robust, _ = cached_fit("robust", robust, lambda m: fit_classifier(
        m, Xn[adv_idx], Xc[adv_idx], y[adv_idx], Vn, Vc, vy, epochs=E["adv"], lr=3e-4,
        class_weight=cw, patience=3, adv=pgd, log=log))
    c1, a1 = adv_detection(robust)
    R["adversarial"] = {"threat_model": f"targeted PGD-10, eps=1.0 sd (log space) on attacker-controllable features only: {MUTABLE}",
                        "standard": {"clean_detection": c0, "adversarial_detection": a0},
                        "adversarially_trained": {"clean_detection": c1, "adversarial_detection": a1,
                                                  **{k: v for k, v in evaluate(ty, predict_logits(robust, Tn, Tc).argmax(1), sp.test).items()
                                                     if k in ("macro_f1", "detection_rate", "false_positive_rate")}}}
    log("adversarial", R["adversarial"])

    torch.save({"state_dict": best.state_dict(), "robust_state_dict": robust.state_dict(),
                "cards": pre.cardinalities, "pre": pre.__dict__, "novelty_threshold": thr,
                "det_mu": det.mu, "det_prec": det.prec, "det_classes": det.classes, "val_scores": sv},
               ROOT / "artifacts" / "ids_model.pt")

    # ---------------- SOC agent evaluation ----------------
    R["agent"] = evaluate_agent(sp, best, det, thr, sv, Tn, Tc, pred, n=60 if args.quick else 400)
    log("agent", {k: v for k, v in R["agent"].items() if k != "examples"})

    plots(R)
    (ROOT / "results" / "metrics.json").write_text(json.dumps(R, indent=2, default=float))
    (ROOT / "results" / "sample_incident_reports.json").write_text(json.dumps(R["agent"]["examples"], indent=2, default=float))


def build_alert_stream(test: pd.DataFrame, Tn, Tc, pred, n: int, seed: int = 0) -> list[Alert]:
    """Synthesise SOC metadata (IPs, timestamps) around real NSL-KDD records.

    NSL-KDD has no IPs/timestamps, so attacks are grouped into simulated campaigns (one source
    per attack type) to exercise correlation logic. Clearly marked as simulated in the README."""
    rng = np.random.default_rng(seed)
    flagged = np.where(pred != 0)[0]
    idx = rng.choice(flagged, min(n, len(flagged)), replace=False)
    src = {a: f"203.0.113.{i + 10}" for i, a in enumerate(sorted(test["attack"].unique()))}  # TEST-NET-3
    alerts = []
    for j, i in enumerate(idx):
        r = test.iloc[i]
        alerts.append(Alert(f"ALRT-{j:05d}", float(rng.uniform(0, 3600)),
                            src[r["attack"]] if r["attack"] != "normal" else f"10.20.{rng.integers(0, 255)}.{rng.integers(1, 255)}",
                            f"10.0.{rng.integers(0, 4)}.{rng.integers(1, 60)}", r["service"], Tn[i], Tc[i],
                            raw={c: r[c] for c in NUMERIC + CATEGORICAL} | {"_true_attack": r["attack"]}))
    return alerts


def evaluate_agent(sp, model, det, thr, sv, Tn, Tc, pred, n: int) -> dict:
    alerts = build_alert_stream(sp.test, Tn, Tc, pred, n)
    ctx = TriageContext(model, det, thr, sv, alerts)
    outs, t0 = [], time.time()
    for a in alerts:
        outs.append(triage(a, ctx))
    dt = (time.time() - t0) / len(alerts)
    valid = [o for o in outs if not o["validation_errors"]]
    hits, n_att, sev_counts = 0, 0, {}
    for a, o in zip(alerts, outs):
        sev_counts[o["report"]["severity"]] = sev_counts.get(o["report"]["severity"], 0) + 1
        true = a.raw["_true_attack"]
        if true == "normal":
            continue
        n_att += 1
        if set(o["report"]["techniques"]) & set(ATTACK_TECHNIQUES.get(true, [])):
            hits += 1
    auto_exec = sum(1 for o in outs for p in o["pending_actions"] if p["status"] != "PENDING_HUMAN_APPROVAL")
    ex = [o for o in outs if o["report"]["severity"] in ("critical", "high")][:3] + \
         [o for o in outs if "UNKNOWN" in o["report"]["summary"]][:2]
    for o in ex:
        o["true_attack_type (hidden from agent)"] = next(a.raw["_true_attack"] for a in alerts if a.alert_id == o["alert_id"])
    return {"alerts_triaged": len(outs), "reports_passing_validation": len(valid) / len(outs),
            "attack_technique_hit_rate": hits / max(1, n_att),
            "actions_auto_executed": auto_exec, "mean_seconds_per_alert": dt,
            "severity_distribution": sev_counts, "examples": ex}


def plots(R):
    m = R["models"]
    names = ["xgboost", "mlp", "ft_transformer", "ft_transformer_ssl"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(names, [m[k]["macro_f1"] for k in names], color=["#999", "#999", "#5b8bd6", "#1f4e9c"])
    ax[0].set_title("Macro-F1 on KDDTest+ (5 classes)"); ax[0].set_ylim(0, 1); ax[0].tick_params(axis="x", rotation=15)
    ax[1].bar(names, [m[k]["novel_attack_detection_rate"] for k in names], color=["#999", "#999", "#5b8bd6", "#1f4e9c"])
    z = R["zero_day"]["novel_detected_classifier_plus_novelty"]
    ax[1].axhline(z, color="#d1495b", ls="--", label=f"deployed model + novelty detector: {z:.0%}")
    ax[1].set_title("Detection rate on 17 never-seen attack types"); ax[1].set_ylim(0, 1)
    ax[1].legend(); ax[1].tick_params(axis="x", rotation=15)
    fig.tight_layout(); fig.savefig(FIG / "model_comparison.png", dpi=140); plt.close(fig)

    le = R["label_efficiency"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, c in [("xgboost", "#999"), ("ft_scratch", "#5b8bd6"), ("ft_ssl_pretrained", "#1f4e9c")]:
        ax.plot(list(le), [le[f][k] for f in le], "o-", color=c, label=k)
    ax.set_xlabel("Fraction of labelled training data"); ax.set_ylabel("Test macro-F1")
    ax.set_title("Label efficiency: value of self-supervised pre-training"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "label_efficiency.png", dpi=140); plt.close(fig)

    a = R["adversarial"]
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = np.arange(2)
    ax.bar(xs - 0.18, [a["standard"]["clean_detection"], a["adversarially_trained"]["clean_detection"]], 0.36, label="clean", color="#5b8bd6")
    ax.bar(xs + 0.18, [a["standard"]["adversarial_detection"], a["adversarially_trained"]["adversarial_detection"]], 0.36, label="under PGD evasion", color="#d1495b")
    ax.set_xticks(xs, ["standard training", "adversarial training"]); ax.set_ylim(0, 1)
    ax.set_ylabel("Attack detection rate"); ax.set_title("Robustness to feature-constrained evasion"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "adversarial.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
