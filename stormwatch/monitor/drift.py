"""
StormWatch AI - Data Drift Detection
Monitors feature distributions for drift using statistical tests.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from stormwatch.logger import get_logger

log = get_logger(__name__)

DB_PATH = Path(__file__).parent.parent.parent / "mlflow" / "monitor.db"
REFERENCE_WINDOW = 500  # samples to keep in reference
ALERT_THRESHOLD = 0.05  # p-value below this triggers drift alert

# ──────────────────────────────────────────────
#  Data types
# ──────────────────────────────────────────────


@dataclass
class DriftResult:
    """Result of a single feature drift test."""

    feature: str
    statistic: float
    p_value: float
    drifted: bool
    reference_mean: float
    current_mean: float
    reference_std: float
    current_std: float


@dataclass
class DriftReport:
    """Full drift report for a model."""

    model_name: str
    timestamp: str
    total_features: int
    drifted_features: int
    drift_score: float  # fraction of features drifted
    features: List[DriftResult]
    alert: bool


# ──────────────────────────────────────────────
#  Database helpers
# ──────────────────────────────────────────────


def _get_db() -> sqlite3.Connection:
    """Get or create the monitoring SQLite database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            features TEXT NOT NULL,
            prediction TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reference_stats (
            model_name TEXT PRIMARY KEY,
            feature_name TEXT NOT NULL,
            mean REAL NOT NULL,
            std REAL NOT NULL,
            count INTEGER NOT NULL
        )
    """)
    conn.commit()
    return conn


def record_prediction(
    model_name: str, features: Dict[str, Any], prediction: Any
) -> None:
    """Record a prediction for drift monitoring."""
    conn = _get_db()
    conn.execute(
        "INSERT INTO predictions (model_name, timestamp, features, prediction) VALUES (?, ?, ?, ?)",
        (
            model_name,
            datetime.now(timezone.utc).isoformat(),
            str(features),
            str(prediction),
        ),
    )
    conn.commit()
    conn.close()


# ──────────────────────────────────────────────
#  Drift detection
# ──────────────────────────────────────────────


def compute_drift(reference: pd.DataFrame, current: pd.DataFrame) -> List[DriftResult]:
    """Run Kolmogorov-Smirnov test on each numeric feature.

    Args:
        reference: Reference distribution DataFrame.
        current: Recent predictions DataFrame.

    Returns:
        List of DriftResult per feature.
    """
    results: List[DriftResult] = []

    for col in reference.select_dtypes(include=[np.number]).columns:
        if col not in current.columns:
            continue

        ref_vals = reference[col].dropna().values
        cur_vals = current[col].dropna().values

        if len(ref_vals) < 10 or len(cur_vals) < 10:
            continue

        stat, p_value = ks_2samp(ref_vals, cur_vals, alternative="two-sided")

        results.append(
            DriftResult(
                feature=col,
                statistic=float(stat),
                p_value=float(p_value),
                drifted=bool(p_value < ALERT_THRESHOLD),
                reference_mean=float(ref_vals.mean()),
                current_mean=float(cur_vals.mean()),
                reference_std=float(ref_vals.std(ddof=0)),
                current_std=float(cur_vals.std(ddof=0)),
            )
        )

    return results


def run_drift_check(model_name: str, model: Optional[object] = None) -> Dict[str, Any]:
    """Run a full drift check for a model.

    Loads recent predictions from the monitor database and compares
    the last N predictions against the stored reference distribution.

    Args:
        model_name: Name of the model to check.
        model: Optional model instance (unused, for interface compatibility).

    Returns:
        Drift report as a dict.
    """
    conn = _get_db()
    cursor = conn.execute(
        "SELECT features FROM predictions WHERE model_name = ? ORDER BY id DESC LIMIT ?",
        (model_name, REFERENCE_WINDOW),
    )
    rows = cursor.fetchall()
    conn.close()

    if len(rows) < 20:
        return {
            "model": model_name,
            "status": "insufficient_data",
            "samples": len(rows),
            "message": f"Need at least 20 samples, have {len(rows)}",
            "features": [],
            "drifted_features": 0,
            "drift_score": 0.0,
            "alert": False,
        }

    # Parse features from stored strings
    features_list = []
    for (feat_str,) in rows:
        try:
            feat_dict = eval(feat_str)  # safe since we wrote it
            features_list.append(feat_dict)
        except Exception:
            continue

    df = pd.DataFrame(features_list)

    # Split into reference (older 2/3) and current (recent 1/3)
    split = int(len(df) * 2 / 3)
    reference = df.iloc[:split]
    current = df.iloc[split:]

    results = compute_drift(reference, current)
    drifted = [r for r in results if r.drifted]

    report = DriftReport(
        model_name=model_name,
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_features=len(results),
        drifted_features=len(drifted),
        drift_score=len(drifted) / max(len(results), 1),
        features=results,
        alert=len(drifted) >= max(2, len(results) // 3),
    )

    if report.alert:
        log.warning(
            "Drift alert for %s: %d/%d features drifted (score=%.2f)",
            model_name,
            report.drifted_features,
            report.total_features,
            report.drift_score,
        )

    return {
        "model": report.model_name,
        "status": "ok",
        "samples": len(rows),
        "timestamp": report.timestamp,
        "total_features": report.total_features,
        "drifted_features": report.drifted_features,
        "drift_score": round(report.drift_score, 3),
        "features": [
            {
                "feature": f.feature,
                "statistic": round(f.statistic, 4),
                "p_value": round(f.p_value, 4),
                "drifted": f.drifted,
                "reference_mean": round(f.reference_mean, 2),
                "current_mean": round(f.current_mean, 2),
            }
            for f in results
        ],
        "alert": report.alert,
    }
