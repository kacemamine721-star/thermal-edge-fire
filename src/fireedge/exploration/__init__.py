"""Exploration and statistics package."""
from .band_stats import ChannelAccumulator, collect_band_stats, correlation_matrix
from .separability import (
    fisher_ratio,
    auc_score,
    kolmogorov_smirnov,
    band_separability_report,
)

__all__ = [
    "ChannelAccumulator",
    "collect_band_stats",
    "correlation_matrix",
    "fisher_ratio",
    "auc_score",
    "kolmogorov_smirnov",
    "band_separability_report",
]
