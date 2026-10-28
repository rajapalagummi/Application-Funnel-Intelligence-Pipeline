import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from credit_risk.scorecard import fit_numeric, fit_categorical, to_woe, is_monotonic, to_score, discrimination, score_params
from credit_risk.validation import psi, csi, calibration
from credit_risk.report_agent import numbers_in, supported


def sample(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    y = (rng.random(n) < 1 / (1 + np.exp(4 * (x - 0.3)))).astype(int)
    return pd.Series(x), pd.Series(y)


def test_numeric_woe_is_monotonic():
    x, y = sample()
    spec = fit_numeric(x, y)
    assert is_monotonic(spec)
    assert spec["table"]["iv"].sum() > 0.1


def test_missing_gets_own_bin():
    x, y = sample()
    x[:500] = np.nan
    spec = fit_numeric(x, y)
    assert -1 in spec["woe"]


def test_binary_feature_is_binned():
    x, y = sample()
    spec = fit_numeric(pd.Series((x < 0.3).astype(int)), y)
    assert spec is not None and len(spec["table"]) == 2


def test_rare_categories_grouped():
    x = pd.Series(["A"] * 600 + ["B"] * 395 + ["C"] * 5)
    y = pd.Series([0, 1] * 500)
    spec = fit_categorical(x, y)
    assert "C" not in spec["levels"]
    assert "OTHER" in spec["woe"]


def test_unseen_values_map_to_zero_woe():
    x, y = sample()
    spec = {"f": fit_categorical(pd.Series(["A", "B"] * 2500), y)}
    w = to_woe(pd.DataFrame({"f": ["Z"]}), spec, ["f"])
    assert w["f"].iloc[0] == 0.0


def test_score_anchor():
    factor, offset = score_params()
    p = 1 / 51
    assert abs(to_score(np.array([p]))[0] - 600) < 1e-6
    assert abs(to_score(np.array([1 / 101]))[0] - to_score(np.array([p]))[0] - 20) < 0.5


def test_discrimination_perfect():
    d = discrimination(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert d["auc"] == 1.0 and d["gini"] == 1.0 and d["ks"] == 1.0


def test_psi_zero_for_same_distribution():
    a = np.random.default_rng(1).normal(size=10000)
    edges = np.r_[-np.inf, np.quantile(a, np.linspace(0.1, 0.9, 9)), np.inf]
    assert psi(a, a, edges) < 1e-9


def test_csi_detects_shift():
    train = pd.Series(["a"] * 900 + ["b"] * 100)
    test = pd.Series(["a"] * 500 + ["b"] * 500)
    assert csi(train, test) > 0.1


def test_calibration_perfect_model():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.01, 0.5, 20000)
    y = (rng.random(20000) < p).astype(int)
    table, hl, p_value = calibration(y, p)
    assert len(table) == 10 and p_value > 0.01


def test_guardrail_number_matching():
    sources = [0.7412, 0.0234, 61503.0, 0.95]
    assert supported("0.7412", sources)
    assert supported("0.74", sources)
    assert supported("2.34", sources)
    assert supported("61,503", sources)
    assert supported("95", sources)
    assert not supported("0.81", sources)
    assert numbers_in("Top 5 features in 2026 had AUC 0.7412") == ["0.7412"]
