"""FT-Transformer (Gorishniy et al., NeurIPS 2021) for network-connection records,
with a masked-feature-modelling head for self-supervised pre-training.

Every feature (numeric or categorical) becomes a token; a transformer encoder then models
interactions between features (e.g. *high SYN-error rate* x *many connections to the same
service* -> SYN flood). The [CLS] embedding feeds the classifier and doubles as the
representation used for novelty detection.
"""
from __future__ import annotations

import torch
from torch import nn


class FeatureTokenizer(nn.Module):
    def __init__(self, n_num: int, cards: list[int], d: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_num, d) * 0.02)
        self.b = nn.Parameter(torch.zeros(n_num, d))
        self.cat = nn.ModuleList(nn.Embedding(c, d) for c in cards)
        self.cat_bias = nn.Parameter(torch.zeros(len(cards), d))
        self.mask_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)

    def forward(self, x_num, x_cat, mask=None):
        t_num = x_num.unsqueeze(-1) * self.w + self.b                               # (B, N, d)
        t_cat = torch.stack([e(x_cat[:, i]) for i, e in enumerate(self.cat)], 1) + self.cat_bias
        tok = torch.cat([t_num, t_cat], 1)
        if mask is not None:                                                         # (B, N+C) bool
            tok = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(tok), tok)
        return tok


class FTTransformer(nn.Module):
    def __init__(self, n_num: int, cards: list[int], n_classes: int = 5, d: int = 32,
                 layers: int = 2, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_num, self.cards = n_num, cards
        self.tok = FeatureTokenizer(n_num, cards, d)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, n_classes))
        # self-supervised heads: reconstruct each masked numeric value / categorical code
        self.rec_num = nn.Linear(d, 1)
        self.rec_cat = nn.ModuleList(nn.Linear(d, c) for c in cards)

    def encode(self, x_num, x_cat, mask=None):
        t = self.tok(x_num, x_cat, mask)
        h = self.encoder(torch.cat([self.cls.expand(len(t), -1, -1), t], 1))
        return self.norm(h)                                                          # (B, 1+N+C, d)

    def embed(self, x_num, x_cat):
        return self.encode(x_num, x_cat)[:, 0]

    def forward(self, x_num, x_cat):
        return self.head(self.embed(x_num, x_cat))

    def mfm_loss(self, x_num, x_cat, mask_rate: float = 0.3):
        """Masked feature modelling: hide 30% of feature tokens, predict them from the rest."""
        b, n_tok = len(x_num), self.n_num + len(self.cards)
        mask = torch.rand(b, n_tok, device=x_num.device) < mask_rate
        h = self.encode(x_num, x_cat, mask)[:, 1:]
        m_num, m_cat = mask[:, : self.n_num], mask[:, self.n_num:]
        pred_num = self.rec_num(h[:, : self.n_num]).squeeze(-1)
        loss = ((pred_num - x_num) ** 2)[m_num].mean() if m_num.any() else 0.0
        for i, head in enumerate(self.rec_cat):
            sel = m_cat[:, i]
            if sel.any():
                loss = loss + nn.functional.cross_entropy(head(h[sel, self.n_num + i]), x_cat[sel, i]) / len(self.cards)
        return loss


class MLP(nn.Module):
    """Strong tabular deep baseline (categoricals embedded)."""
    def __init__(self, n_num: int, cards: list[int], n_classes: int = 5, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.emb = nn.ModuleList(nn.Embedding(c, 8) for c in cards)
        self.net = nn.Sequential(nn.Linear(n_num + 8 * len(cards), hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden),
                                 nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Linear(hidden, n_classes)

    def embed(self, x_num, x_cat):
        e = torch.cat([m(x_cat[:, i]) for i, m in enumerate(self.emb)], 1)
        return self.net(torch.cat([x_num, e], 1))

    def forward(self, x_num, x_cat):
        return self.head(self.embed(x_num, x_cat))
