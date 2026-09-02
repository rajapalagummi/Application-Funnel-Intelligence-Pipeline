import duckdb
import numpy as np
import pandas as pd
from scipy import stats

DB_PATH = "data/funneliq.duckdb"


def load_did_data(con):
    return con.execute("""
        SELECT
            SK_ID_PREV,
            policy_group,
            final_stage,
            days_since_decision,
            CASE WHEN final_stage = 5 THEN 1 ELSE 0 END AS converted,
            CASE WHEN final_stage >= 3 THEN 1 ELSE 0 END AS submitted,
            CASE WHEN days_since_decision > 500 THEN 'post' ELSE 'pre' END AS period,
            NAME_CLIENT_TYPE,
            AMT_APPLICATION
        FROM funnel_stages
        WHERE days_since_decision IS NOT NULL
    """).df()


def run_did(df, outcome, treatment_col="policy_group",
            treatment_val="new_policy", period_col="period"):
    treat_post = df[(df[treatment_col] == treatment_val) & (df[period_col] == "post")][outcome].mean()
    treat_pre = df[(df[treatment_col] == treatment_val) & (df[period_col] == "pre")][outcome].mean()
    ctrl_post = df[(df[treatment_col] != treatment_val) & (df[period_col] == "post")][outcome].mean()
    ctrl_pre = df[(df[treatment_col] != treatment_val) & (df[period_col] == "pre")][outcome].mean()

    did = (treat_post - treat_pre) - (ctrl_post - ctrl_pre)

    treat = df[df[treatment_col] == treatment_val][outcome]
    ctrl = df[df[treatment_col] != treatment_val][outcome]
    se = np.sqrt(treat.var() / len(treat) + ctrl.var() / len(ctrl))
    t_stat = did / (se + 1e-9)
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    ci_lower = did - 1.96 * se
    ci_upper = did + 1.96 * se

    return {
        "outcome": outcome,
        "did_estimate": round(did, 6),
        "se": round(se, 6),
        "t_stat": round(t_stat, 4),
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "treat_pre": round(treat_pre, 4),
        "treat_post": round(treat_post, 4),
        "ctrl_pre": round(ctrl_pre, 4),
        "ctrl_post": round(ctrl_post, 4),
    }


def test_parallel_trends(df, outcome, treatment_col="policy_group",
                          treatment_val="new_policy"):
    pre_df = df[df["period"] == "pre"].copy()
    pre_df["time_bin"] = pd.qcut(pre_df["days_since_decision"], q=4, labels=False)

    treat_trend = pre_df[pre_df[treatment_col] == treatment_val].groupby("time_bin")[outcome].mean()
    ctrl_trend = pre_df[pre_df[treatment_col] != treatment_val].groupby("time_bin")[outcome].mean()

    n = min(len(treat_trend), len(ctrl_trend))
    if n < 3:
        return {"p_value": None, "satisfied": None}

    t, p = stats.ttest_ind(np.diff(treat_trend.values[:n]), np.diff(ctrl_trend.values[:n]))
    return {"p_value": round(float(p), 4), "satisfied": bool(p > 0.05)}


def run_placebo_test(df, outcome, treatment_col="policy_group"):
    df_placebo = df[df["period"] == "pre"].copy()
    midpoint = df_placebo["days_since_decision"].median()
    df_placebo["period"] = np.where(
        df_placebo["days_since_decision"] > midpoint, "post", "pre"
    )
    result = run_did(df_placebo, outcome, treatment_col)
    result["test_type"] = "placebo"
    return result


def run_module2():
    con = duckdb.connect(DB_PATH)
    df = load_did_data(con)
    con.close()

    print("\n=== Module 2: Causal Intervention Analysis ===")
    results = {}

    for outcome in ["converted", "submitted"]:
        parallel = test_parallel_trends(df, outcome)
        did = run_did(df, outcome)
        placebo = run_placebo_test(df, outcome)

        results[outcome] = {
            "parallel_trends": parallel,
            "did": did,
            "placebo": placebo,
        }

        print(f"\nOutcome: {outcome}")
        print(f"  Parallel trends: p={parallel['p_value']}, satisfied={parallel['satisfied']}")
        print(f"  DiD estimate: {did['did_estimate']} (p={did['p_value']}, sig={did['significant']})")
        print(f"  95% CI: [{did['ci_lower']}, {did['ci_upper']}]")
        print(f"  Placebo test: {placebo['did_estimate']} (p={placebo['p_value']}) — should be non-significant")

    return results
