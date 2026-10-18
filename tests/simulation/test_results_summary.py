"""Tests for rc.results_summary. Run with: python -m pytest tests/simulation/test_results_summary.py"""
import numpy as np
import pandas as pd
import pytest

import rfcorr as rc


def test_results_summary_no_group():
    results = pd.DataFrame({
        "bias": [0.1, -0.1], "est": [0.5, 0.3], "se": [0.1, 0.3],
        "cover": [1, 0], "cover_raw": [1, 1], "reject": [1, 1],
        "reject_raw": [0, 1], "runtime": [1.0, 3.0],
    })
    out = rc.results_summary(results)
    assert out.shape[0] == 1
    row = out.iloc[0]
    assert row["bias_mean"] == pytest.approx(0.0)
    assert row["est_sd"] == pytest.approx(np.std([0.5, 0.3], ddof=1))
    assert row["se_mean"] == pytest.approx(0.2)
    assert row["cover_rate"] == 0.5 and row["cover_rate_raw"] == 1.0
    assert row["power"] == 1.0 and row["power_raw"] == 0.5
    assert row["runtime_mean"] == 2.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
