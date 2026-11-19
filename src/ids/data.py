"""NSL-KDD loading, attack taxonomy and preprocessing.

NSL-KDD (Tavallaee et al., 2009; Canadian Institute for Cybersecurity) is the cleaned
successor of KDD'99. Its official test split is valuable for a *realistic* evaluation:
**17 attack types in KDDTest+ never appear in training** (3,750 records). A model that only
memorises known signatures fails on them - exactly the zero-day problem a federal SOC faces.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "attack", "difficulty",
]
CATEGORICAL = ["protocol_type", "service", "flag"]
NUMERIC = [c for c in COLUMNS[:-2] if c not in CATEGORICAL and c != "num_outbound_cmds"]  # constant col dropped

# Attack -> family (standard NSL-KDD taxonomy, incl. test-only attacks)
FAMILY = {
    "normal": "normal",
    **dict.fromkeys(["back", "land", "neptune", "pod", "smurf", "teardrop", "apache2", "mailbomb",
                     "processtable", "udpstorm"], "DoS"),
    **dict.fromkeys(["ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"], "Probe"),
    **dict.fromkeys(["ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy", "warezclient",
                     "warezmaster", "httptunnel", "named", "sendmail", "snmpgetattack", "snmpguess",
                     "xlock", "xsnoop", "worm"], "R2L"),
    **dict.fromkeys(["buffer_overflow", "loadmodule", "perl", "rootkit", "ps", "sqlattack", "xterm"], "U2R"),
}
CLASSES = ["normal", "DoS", "Probe", "R2L", "U2R"]

# Features an attacker can plausibly manipulate without breaking the attack itself.
# Used to constrain the adversarial-robustness evaluation to *realistic* perturbations.
MUTABLE = ["duration", "src_bytes", "dst_bytes", "hot", "num_file_creations", "num_access_files",
           "count", "srv_count"]


def read_nslkdd(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df["family"] = df["attack"].map(FAMILY).fillna("R2L")
    df["y"] = df["family"].map({c: i for i, c in enumerate(CLASSES)}).astype(int)
    df["is_attack"] = (df["family"] != "normal").astype(int)
    return df


@dataclass
class Preprocessor:
    """log1p + robust standardisation for numerics; integer codes (+UNK) for categoricals."""
    cat_maps: dict
    mu: np.ndarray
    sd: np.ndarray

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "Preprocessor":
        cat_maps = {c: {v: i + 1 for i, v in enumerate(sorted(df[c].unique()))} for c in CATEGORICAL}
        x = np.log1p(df[NUMERIC].to_numpy(np.float64).clip(min=0))
        return cls(cat_maps, x.mean(0), x.std(0) + 1e-6)

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = (np.log1p(df[NUMERIC].to_numpy(np.float64).clip(min=0)) - self.mu) / self.sd
        c = np.stack([df[col].map(m).fillna(0).astype(int).to_numpy() for col, m in self.cat_maps.items()], 1)
        return np.clip(x, -10, 10).astype(np.float32), c.astype(np.int64)

    def inverse_numeric(self, x: np.ndarray) -> np.ndarray:
        return np.expm1(x * self.sd + self.mu)

    @property
    def cardinalities(self) -> list[int]:
        return [len(m) + 1 for m in self.cat_maps.values()]


@dataclass
class Splits:
    pre: Preprocessor
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    known_attacks: set


def load(data_dir: str | Path, val_frac: float = 0.05, seed: int = 0) -> Splits:
    data_dir = Path(data_dir)
    full = read_nslkdd(data_dir / "KDDTrain+.txt")
    test = read_nslkdd(data_dir / "KDDTest+.txt")
    val = full.groupby("family", group_keys=False).sample(frac=val_frac, random_state=seed)
    train = full.drop(val.index)
    known = set(full["attack"])
    test["novel_attack"] = (~test["attack"].isin(known)).astype(int)
    return Splits(Preprocessor.fit(train), train.reset_index(drop=True), val.reset_index(drop=True),
                  test.reset_index(drop=True), known)
