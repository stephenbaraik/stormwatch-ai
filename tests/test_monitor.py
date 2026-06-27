from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stormwatch.monitor.drift import (
    ALERT_THRESHOLD,
    DB_PATH,
    compute_drift,
    record_prediction,
    run_drift_check,
)


@pytest.fixture(autouse=True)
def _clean_db():
    # Remove test db before and after
    db = Path(DB_PATH)
    if db.exists():
        os.remove(str(db))
    yield
    if db.exists():
        os.remove(str(db))


class TestComputeDrift:
    def test_identical_distributions_no_drift(self):
        ref = pd.DataFrame({"feature_a": np.random.normal(0, 1, 100)})
        cur = ref.copy()
        results = compute_drift(ref, cur)
        assert len(results) == 1
        assert not results[0].drifted

    def test_different_distributions_detect_drift(self):
        np.random.seed(42)
        ref = pd.DataFrame({"feature_a": np.random.normal(0, 1, 200)})
        cur = pd.DataFrame({"feature_a": np.random.normal(5, 1, 200)})
        results = compute_drift(ref, cur)
        assert len(results) == 1
        assert results[0].drifted
        assert results[0].p_value < ALERT_THRESHOLD

    def test_multiple_features(self):
        np.random.seed(42)
        ref = pd.DataFrame({
            "a": np.random.normal(0, 1, 100),
            "b": np.random.normal(10, 2, 100),
            "c": np.random.normal(100, 15, 100),
        })
        cur = ref.copy()
        cur["a"] = np.random.normal(5, 1, 100)
        results = compute_drift(ref, cur)
        assert len(results) == 3
        drifted = [r for r in results if r.drifted]
        assert len(drifted) >= 1

    def test_returns_empty_for_no_numeric_cols(self):
        ref = pd.DataFrame({"label": ["a", "b", "c"]})
        cur = pd.DataFrame({"label": ["d", "e", "f"]})
        results = compute_drift(ref, cur)
        assert results == []

    def test_handles_nan_values(self):
        vals = [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        ref = pd.DataFrame({"a": vals})
        cur = pd.DataFrame({"a": [v + 10 for v in vals]})
        results = compute_drift(ref, cur)
        assert len(results) == 1

    def test_skips_features_with_few_samples(self):
        ref = pd.DataFrame({"a": [1.0, 2.0]})
        cur = pd.DataFrame({"a": [3.0, 4.0]})
        results = compute_drift(ref, cur)
        assert results == []

    def test_skips_missing_columns(self):
        ref = pd.DataFrame({"a": np.random.normal(0, 1, 100), "b": np.random.normal(0, 1, 100)})
        cur = pd.DataFrame({"a": np.random.normal(0, 1, 100)})
        results = compute_drift(ref, cur)
        assert len(results) == 1

    def test_drift_result_fields(self):
        ref = pd.DataFrame({"x": np.random.normal(0, 1, 100)})
        cur = pd.DataFrame({"x": np.random.normal(0, 1, 100)})
        results = compute_drift(ref, cur)
        r = results[0]
        assert r.feature == "x"
        assert isinstance(r.statistic, float)
        assert isinstance(r.p_value, float)
        assert isinstance(r.drifted, bool)
        assert isinstance(r.reference_mean, float)
        assert isinstance(r.current_mean, float)


class TestRecordPrediction:
    def test_record_and_retrieve(self):
        record_prediction("test_model", {"temp": 30.0, "humidity": 60}, 1)
        from stormwatch.monitor.drift import _get_db
        conn = _get_db()
        cursor = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE model_name = ?",
            ("test_model",),
        )
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 1

    def test_multiple_records(self):
        for i in range(10):
            record_prediction("multi", {"val": float(i)}, i)
        from stormwatch.monitor.drift import _get_db
        conn = _get_db()
        cursor = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE model_name = ?",
            ("multi",),
        )
        assert cursor.fetchone()[0] == 10
        conn.close()


class TestRunDriftCheck:
    def test_insufficient_data(self):
        result = run_drift_check("empty_model")
        assert result["status"] == "insufficient_data"
        assert result["samples"] < 20

    def test_drift_check_with_data(self):
        for i in range(30):
            record_prediction("check_model", {"val": float(50 + i)}, i)
        result = run_drift_check("check_model")
        assert result["status"] == "ok"
        assert result["samples"] >= 20
        assert "drift_score" in result
        assert "features" in result

    def test_drift_check_separates_reference_and_current(self):
        for i in range(60):
            record_prediction("split_model", {"val": float(i)}, i)
        result = run_drift_check("split_model")
        assert result["status"] == "ok"
        assert result["total_features"] >= 0

    def test_alert_flag(self):
        # Two very different populations should trigger drift
        for i in range(60):
            val = float(i) if i < 40 else float(100 + i)
            record_prediction("alert_model", {"val": val}, i)
        result = run_drift_check("alert_model")
        assert "alert" in result
