"""Thermal band active-fire separability metrics."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


def fisher_ratio(fire_vals: np.ndarray, bg_vals: np.ndarray) -> float:
    """Compute Fisher's Discriminant Ratio between fire and background pixels.
    
    FDR = (mu_fire - mu_bg)^2 / (var_fire + var_bg)
    Higher means better inherent class separability.
    """
    f = np.asarray(fire_vals, dtype=np.float64)
    b = np.asarray(bg_vals, dtype=np.float64)
    if f.size == 0 or b.size == 0:
        return 0.0

    mu_f, var_f = float(f.mean()), float(f.var())
    mu_b, var_b = float(b.mean()), float(b.var())

    denom = var_f + var_b
    if denom <= 1e-9:
        return 0.0
    return float(((mu_f - mu_b) ** 2) / denom)


def auc_score(fire_vals: np.ndarray, bg_vals: np.ndarray) -> float:
    """Compute ROC-AUC score for single-band intensity thresholding."""
    f = np.asarray(fire_vals, dtype=np.float64)
    b = np.asarray(bg_vals, dtype=np.float64)
    if f.size == 0 or b.size == 0:
        return 0.5

    # Subsample if large to keep computation fast
    max_pts = 10_000
    if f.size > max_pts:
        f = np.random.choice(f, max_pts, replace=False)
    if b.size > max_pts:
        b = np.random.choice(b, max_pts, replace=False)

    y_true = np.concatenate([np.ones(f.size), np.zeros(b.size)])
    y_score = np.concatenate([f, b])

    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.5


def kolmogorov_smirnov(fire_vals: np.ndarray, bg_vals: np.ndarray) -> Tuple[float, float]:
    """Compute 2-sample Kolmogorov-Smirnov test statistic and p-value."""
    f = np.asarray(fire_vals, dtype=np.float64)
    b = np.asarray(bg_vals, dtype=np.float64)
    if f.size == 0 or b.size == 0:
        return 0.0, 1.0

    res = stats.ks_2samp(f, b)
    return float(res.statistic), float(res.pvalue)


def band_separability_report(
    channel_samples: Dict[str, Tuple[np.ndarray, np.ndarray]]
) -> pd.DataFrame:
    """Generate separability report given dict of band_name -> (fire_vals, bg_vals)."""
    rows: List[Dict] = []
    for band_name, (f_vals, b_vals) in channel_samples.items():
        fdr = fisher_ratio(f_vals, b_vals)
        auc = auc_score(f_vals, b_vals)
        ks_stat, ks_p = kolmogorov_smirnov(f_vals, b_vals)

        rows.append(
            {
                "band": band_name,
                "fire_pixels": len(f_vals),
                "bg_pixels": len(b_vals),
                "fisher_ratio": round(fdr, 4),
                "auc_score": round(auc, 4),
                "ks_statistic": round(ks_stat, 4),
                "ks_pvalue": ks_p,
            }
        )

    return pd.DataFrame(rows).sort_values(by="fisher_ratio", ascending=False)
