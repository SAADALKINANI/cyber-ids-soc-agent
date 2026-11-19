"""Training: self-supervised pre-training, supervised fine-tuning and adversarial training."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _loader(xn, xc, y=None, bs=512, shuffle=True):
    tensors = [torch.from_numpy(xn), torch.from_numpy(xc)] + ([torch.from_numpy(y)] if y is not None else [])
    return DataLoader(TensorDataset(*tensors), batch_size=bs, shuffle=shuffle)


@torch.no_grad()
def predict_logits(model, xn, xc, bs=4096) -> np.ndarray:
    model.eval()
    return torch.cat([model(torch.from_numpy(xn[i:i + bs]), torch.from_numpy(xc[i:i + bs]))
                      for i in range(0, len(xn), bs)]).numpy()


@torch.no_grad()
def embed(model, xn, xc, bs=4096) -> np.ndarray:
    model.eval()
    return torch.cat([model.embed(torch.from_numpy(xn[i:i + bs]), torch.from_numpy(xc[i:i + bs]))
                      for i in range(0, len(xn), bs)]).numpy()


def pretrain_mfm(model, xn, xc, epochs=4, lr=1e-3, log=print) -> list[float]:
    """Masked-feature-modelling pre-training on *unlabelled* traffic."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    hist = []
    for ep in range(epochs):
        model.train(); tot, n = 0.0, 0
        for xb, cb in _loader(xn, xc, bs=512):
            opt.zero_grad(); loss = model.mfm_loss(xb, cb); loss.backward(); opt.step()
            tot += loss.item() * len(xb); n += len(xb)
        hist.append(tot / n); log(f"  pretrain epoch {ep}: masked-feature loss {tot / n:.4f}")
    return hist


def fit_classifier(model, xn, xc, y, vxn, vxc, vy, epochs=12, lr=1e-3, class_weight=None,
                   patience=4, adv=None, log=print):
    """Supervised training with macro-F1 early stopping.

    adv: optional callable (model, xn, xc, y) -> perturbed xn, used for adversarial training
         (half of each batch is replaced by its PGD-perturbed version).
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    w = torch.tensor(class_weight, dtype=torch.float32) if class_weight is not None else None
    loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    best, best_state, bad, t0 = -1, None, 0, time.time()
    for ep in range(epochs):
        model.train()
        for xb, cb, yb in _loader(xn, xc, y, bs=512):
            if adv is not None:
                h = len(xb) // 2
                xb = xb.clone(); xb[:h] = adv(model, xb[:h], cb[:h], yb[:h]); model.train()
            opt.zero_grad(); loss = loss_fn(model(xb, cb), yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        f1 = f1_score(vy, predict_logits(model, vxn, vxc).argmax(1), average="macro")
        log(f"  epoch {ep}: val macro-F1 {f1:.4f}")
        if f1 > best + 1e-4:
            best, best_state, bad = f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, {"best_val_macro_f1": best, "train_seconds": time.time() - t0}


def balanced_weights(y: np.ndarray, n_classes: int, power: float = 0.5) -> np.ndarray:
    """Soft inverse-frequency weights (sqrt), so U2R (52 training rows!) is not ignored
    without letting it dominate the loss."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = (counts.sum() / (counts + 1)) ** power
    return w / w.mean()
