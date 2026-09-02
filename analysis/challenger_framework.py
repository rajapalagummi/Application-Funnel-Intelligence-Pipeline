import numpy as np
import pandas as pd
from scipy import stats


def compute_mde(baseline_rate, alpha=0.05, power=0.80, n_per_group=None):
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    if n_per_group:
        se = np.sqrt(2 * baseline_rate * (1 - baseline_rate) / n_per_group)
        mde = (z_alpha + z_beta) * se
        return round(mde, 4)

    effects = np.arange(0.001, 0.10, 0.001)
    required_n = []
    for effect in effects:
        p1 = baseline_rate
        p2 = baseline_rate + effect
        pooled = (p1 + p2) / 2
        n = (z_alpha + z_beta) ** 2 * 2 * pooled * (1 - pooled) / (effect ** 2)
        required_n.append(int(np.ceil(n)))

    return pd.DataFrame({"effect_size": effects.round(3), "required_n_per_group": required_n})


def monte_carlo_power(baseline_rate, effect_size, n_per_group,
                       n_simulations=1000, alpha=0.05):
    detected = 0
    estimates = []

    for _ in range(n_simulations):
        control = np.random.binomial(1, baseline_rate, n_per_group)
        treatment = np.random.binomial(1, baseline_rate + effect_size, n_per_group)
        t_stat, p_value = stats.ttest_ind(treatment, control)
        if p_value < alpha:
            detected += 1
        estimates.append(treatment.mean() - control.mean())

    power = detected / n_simulations
    mean_est = np.mean(estimates)
    ci_lower = np.percentile(estimates, 2.5)
    ci_upper = np.percentile(estimates, 97.5)

    return {
        "baseline_rate": baseline_rate,
        "effect_size": effect_size,
        "n_per_group": n_per_group,
        "n_simulations": n_simulations,
        "statistical_power": round(power, 4),
        "mean_estimated_effect": round(mean_est, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
    }


def champion_challenger_simulation(X, y, champion_model, challenger_model,
                                    feature_cols, traffic_split=0.20,
                                    n_rounds=100):
    n = len(X)
    champion_aucs = []
    challenger_aucs = []
    conversion_lifts = []

    from sklearn.metrics import roc_auc_score

    for _ in range(n_rounds):
        challenger_idx = np.random.choice(n, size=int(n * traffic_split), replace=False)
        champion_idx = np.setdiff1d(np.arange(n), challenger_idx)

        X_champ = X[champion_idx]
        y_champ = y[champion_idx]
        X_chall = X[challenger_idx]
        y_chall = y[challenger_idx]

        if len(np.unique(y_champ)) > 1:
            champ_auc = roc_auc_score(y_champ, champion_model.predict_proba(X_champ)[:, 1])
            champion_aucs.append(champ_auc)

        if len(np.unique(y_chall)) > 1:
            chall_auc = roc_auc_score(y_chall, challenger_model.predict_proba(X_chall)[:, 1])
            challenger_aucs.append(chall_auc)

        champ_conv = 1 - champion_model.predict_proba(X_champ)[:, 1].mean()
        chall_conv = 1 - challenger_model.predict_proba(X_chall)[:, 1].mean()
        conversion_lifts.append(chall_conv - champ_conv)

    t_stat, p_value = stats.ttest_ind(challenger_aucs, champion_aucs)
    cohen_d = (np.mean(challenger_aucs) - np.mean(champion_aucs)) / (np.std(champion_aucs) + 1e-9)

    return {
        "champion_mean_auc": round(np.mean(champion_aucs), 4),
        "challenger_mean_auc": round(np.mean(challenger_aucs), 4),
        "auc_lift": round(np.mean(challenger_aucs) - np.mean(champion_aucs), 4),
        "p_value": round(float(p_value), 4),
        "significant": p_value < 0.05,
        "cohens_d": round(float(cohen_d), 4),
        "mean_conversion_lift": round(np.mean(conversion_lifts), 4),
        "traffic_split": traffic_split,
        "n_rounds": n_rounds,
    }


def run_module4(model_results):
    print("\n=== Module 4: Champion/Challenger & Power Analysis ===")

    baseline_rate = float(model_results["y"].mean())
    baseline_conversion = 1 - baseline_rate

    print(f"Baseline drop-off rate: {baseline_rate:.2%}")
    print(f"Baseline conversion rate: {baseline_conversion:.2%}")

    power_result = monte_carlo_power(
        baseline_rate=baseline_rate,
        effect_size=0.02,
        n_per_group=5000,
        n_simulations=1000,
    )
    print(f"\nPower analysis (2% MDE, n=5000/group):")
    print(f"  Statistical power: {power_result['statistical_power']:.1%}")
    print(f"  Mean estimated effect: {power_result['mean_estimated_effect']:.4f}")
    print(f"  95% CI: [{power_result['ci_lower']:.4f}, {power_result['ci_upper']:.4f}]")

    power_curve = []
    for effect in [0.01, 0.02, 0.03, 0.05, 0.08]:
        r = monte_carlo_power(baseline_rate, effect, 5000, n_simulations=500)
        power_curve.append(r)
        print(f"  Effect={effect:.2f}: power={r['statistical_power']:.1%}")

    cc_result = champion_challenger_simulation(
        X=model_results["X"],
        y=model_results["y"],
        champion_model=model_results["models"]["xgb"],
        challenger_model=model_results["models"]["lgb"],
        feature_cols=model_results["features"],
    )
    print(f"\nChampion/Challenger (80/20 split, 100 rounds):")
    print(f"  Champion AUC: {cc_result['champion_mean_auc']}")
    print(f"  Challenger AUC: {cc_result['challenger_mean_auc']}")
    print(f"  AUC lift: {cc_result['auc_lift']} (p={cc_result['p_value']}, sig={cc_result['significant']})")
    print(f"  Cohen's d: {cc_result['cohens_d']}")

    return {
        "power_analysis": power_result,
        "power_curve": power_curve,
        "champion_challenger": cc_result,
    }
