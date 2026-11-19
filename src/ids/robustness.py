"""Novelty (zero-day) detection, drift monitoring and adversarial robustness."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.covariance import LedoitWolf


# ---------------------------------------------------------------------------------------------
# Zero-day / novel-attack detection
# ---------------------------------------------------------------------------------------------
class MahalanobisDetector:
    """Class-conditional Gaussian on the classifier's [CLS] embeddings (Lee et al., NeurIPS 2018).

    score(x) = min_k (z - mu_k)^T Sigma^-1 (z - mu_k)   (shared, shrinkage-estimated covariance)

    Traffic far from *every* known class - benign or malicious - gets a high score. That is the
    signal for "something we have never seen", independent of what the classifier predicts.
    """

    def fit(self, z: np.ndarray, y: np.ndarray) -> "MahalanobisDetector":
        self.classes = np.unique(y)
        self.mu = np.stack([z[y == k].mean(0) for k in self.classes])
        centred = np.concatenate([z[y == k] - self.mu[i] for i, k in enumerate(self.classes)])
        self.prec = LedoitWolf().fit(centred).precision_
        return self

    def score(self, z: np.ndarray) -> np.ndarray:
        d = []
        for m in self.mu:
            c = z - m
            d.append(np.einsum("ij,jk,ik->i", c, self.prec, c))
        return np.min(np.stack(d, 1), 1)


def energy_score(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Negative free energy (Liu et al., NeurIPS 2020); higher = more anomalous."""
    m = logits.max(1, keepdims=True)
    return -(T * (np.log(np.exp((logits - m) / T).sum(1)) + m[:, 0] / T))


# ---------------------------------------------------------------------------------------------
# Drift monitoring
# ---------------------------------------------------------------------------------------------
def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index. Rule of thumb: <0.1 stable, 0.1-0.25 moderate, >0.25 major drift."""
    qs = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        return 0.0
    r = np.histogram(np.clip(ref, qs[0], qs[-1]), qs)[0] / len(ref) + 1e-4
    c = np.histogram(np.clip(cur, qs[0], qs[-1]), qs)[0] / len(cur) + 1e-4
    return float(np.sum((c - r) * np.log(c / r)))


def mmd_rbf(a: np.ndarray, b: np.ndarray, n: int = 1500, seed: int = 0) -> float:
    """Unbiased MMD^2 with an RBF kernel (median heuristic) between embedding samples."""
    rng = np.random.default_rng(seed)
    a = a[rng.choice(len(a), min(n, len(a)), replace=False)]
    b = b[rng.choice(len(b), min(n, len(b)), replace=False)]
    x = np.concatenate([a, b])
    d2 = ((x[:, None] - x[None]) ** 2).sum(-1)
    k = np.exp(-d2 / np.median(d2[d2 > 0]))
    na = len(a)
    kaa, kbb, kab = k[:na, :na], k[na:, na:], k[:na, na:]
    return float((kaa.sum() - np.trace(kaa)) / (na * (na - 1)) + (kbb.sum() - np.trace(kbb)) / (len(b) * (len(b) - 1))
                 - 2 * kab.mean())


# ---------------------------------------------------------------------------------------------
# Adversarial robustness (realistic, feature-constrained evasion)
# ---------------------------------------------------------------------------------------------
def make_pgd(mutable_idx: list[int], eps: float = 0.75, steps: int = 10, normal_class: int = 0,
             allow_decrease: bool = False):
    """Targeted PGD evasion: push *attack* records toward the 'normal' class by changing only
    attacker-controllable features (e.g. padding bytes, adding delays, spreading connections).

    Perturbations live in the standardised log-feature space, so eps = 0.75 means up to
    0.75 standard deviations of log-traffic volume - large enough to matter, small enough that
    the attack still works. By default attackers can only *add* (padding/delay), not remove,
    which keeps the attack semantically valid.
    """
    m = torch.zeros(1)

    def attack(model, xn, xc, y):
        nonlocal m
        if m.numel() == 1:
            m = torch.zeros(xn.shape[1]); m[mutable_idx] = 1.0
        model.eval()
        x0 = xn.detach(); x = x0.clone()
        target = torch.full((len(x),), normal_class, dtype=torch.long)
        alpha = 2.5 * eps / steps
        for _ in range(steps):
            x.requires_grad_(True)
            loss = torch.nn.functional.cross_entropy(model(x, xc), target)
            g, = torch.autograd.grad(loss, x)
            with torch.no_grad():
                x = x - alpha * g.sign() * m                     # minimise loss towards 'normal'
                delta = (x - x0).clamp(-eps if allow_decrease else 0.0, eps) * m
                x = x0 + delta
        # benign rows in the batch are left untouched
        keep = (y == normal_class).unsqueeze(1)
        return torch.where(keep, x0, x.detach())
    return attack
