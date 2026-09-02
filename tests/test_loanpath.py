import pytest
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from analysis.causal_analysis import run_did, run_placebo_test
from analysis.monitoring import compute_psi, compute_ks_stat, spc_control_limits
from analysis.challenger_framework import monte_carlo_power, compute_mde


class TestCausalAnalysis:
    def setup_method(self):
        np.random.seed(42)
        n = 2000
        self.df = pd.DataFrame({
            "policy_group": np.random.choice(["new_policy", "old_policy"], n),
            "period": np.random.choice(["pre", "post"], n),
            "converted": np.random.binomial(1, 0.35, n),
            "submitted": np.random.binomial(1, 0.65, n),
            "days_since_decision": np.random.randint(1, 1000, n),
        })

    def test_did_returns_required_fields(self):
        result = run_did(self.df, "converted")
        for field in ["did_estimate", "p_value", "significant", "ci_lower", "ci_upper"]:
            assert field in result

    def test_did_estimate_is_float(self):
        result = run_did(self.df, "converted")
        assert isinstance(result["did_estimate"], float)

    def test_placebo_returns_test_type(self):
        result = run_placebo_test(self.df, "converted")
        assert result["test_type"] == "placebo"


class TestMonitoring:
    def test_psi_zero_for_identical_distributions(self):
        x = np.random.normal(0, 1, 1000)
        psi = compute_psi(x, x)
        assert psi < 0.01

    def test_psi_high_for_different_distributions(self):
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(5, 1, 1000)
        psi = compute_psi(expected, actual)
        assert psi > 0.25

    def test_ks_stat_bounded(self):
        x = np.random.normal(0, 1, 500)
        y = np.random.normal(0, 1, 500)
        ks, p = compute_ks_stat(x, y)
        assert 0 <= ks <= 1
        assert 0 <= p <= 1

    def test_spc_limits_ordering(self):
        series = pd.Series(np.random.uniform(0.3, 0.5, 100))
        mean, ucl, lcl = spc_control_limits(series)
        assert lcl <= mean <= ucl


class TestExperimentation:
    def test_power_between_0_and_1(self):
        result = monte_carlo_power(0.35, 0.05, 1000, n_simulations=100)
        assert 0 <= result["statistical_power"] <= 1

    def test_larger_effect_higher_power(self):
        small = monte_carlo_power(0.35, 0.02, 2000, n_simulations=200)
        large = monte_carlo_power(0.35, 0.10, 2000, n_simulations=200)
        assert large["statistical_power"] >= small["statistical_power"]

    def test_mde_table_returns_dataframe(self):
        result = compute_mde(baseline_rate=0.35)
        assert isinstance(result, pd.DataFrame)
        assert "effect_size" in result.columns
        assert "required_n_per_group" in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
