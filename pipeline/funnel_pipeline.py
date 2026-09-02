import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("data/raw")
DB_PATH = "data/funneliq.duckdb"

STAGE_MAP = {
    "Approved": "offer_accepted",
    "Unused offer": "offer_presented",
    "Refused": "credit_check",
    "Canceled": "verification",
}

FUNNEL_ORDER = [
    "document_submission",
    "identity_verification",
    "credit_check",
    "offer_presentation",
    "offer_acceptance",
]


def load_raw_data(con):
    prev_app_path = RAW_DIR / "previous_application.csv"
    app_train_path = RAW_DIR / "application_train.csv"

    if not prev_app_path.exists():
        print("Home Credit data not found. Run: kaggle datasets download -d home-credit/home-credit-credit-risk-model-stability")
        print("Generating synthetic funnel data for demonstration...")
        return _synthetic_funnel(con)

    con.execute(f"""
        CREATE OR REPLACE TABLE previous_application AS
        SELECT * FROM read_csv_auto('{prev_app_path}')
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE application_train AS
        SELECT * FROM read_csv_auto('{app_train_path}')
    """)
    return True


def _synthetic_funnel(con):
    np.random.seed(42)
    n = 50000
    statuses = np.random.choice(
        ["Approved", "Refused", "Canceled", "Unused offer"],
        size=n, p=[0.35, 0.30, 0.25, 0.10]
    )
    reject_reasons = np.where(
        statuses == "Refused",
        np.random.choice(["HC", "LIMIT", "SCO", "VERIF", "XAP"], size=n),
        "XAP"
    )
    days_decision = np.random.randint(-1000, 0, size=n)
    rate_down = np.random.uniform(0, 0.5, size=n)
    flag_last = np.random.randint(0, 2, size=n)
    hour_appr = np.random.randint(8, 20, size=n)
    client_type = np.random.choice(["New", "Repeater", "Refreshed"], size=n)

    df = pd.DataFrame({
        "SK_ID_PREV": range(n),
        "SK_ID_CURR": np.random.randint(100000, 500000, size=n),
        "NAME_CONTRACT_STATUS": statuses,
        "CODE_REJECT_REASON": reject_reasons,
        "DAYS_DECISION": days_decision,
        "RATE_DOWN_PAYMENT": rate_down,
        "FLAG_LAST_APPL_PER_CONTRACT": flag_last,
        "HOUR_APPR_PROCESS_START": hour_appr,
        "NAME_CLIENT_TYPE": client_type,
        "AMT_APPLICATION": np.random.lognormal(10, 1, size=n),
        "AMT_CREDIT": np.random.lognormal(10, 1, size=n),
        "CNT_PAYMENT": np.random.randint(6, 60, size=n),
    })

    app_df = pd.DataFrame({
        "SK_ID_CURR": range(100000, 150000),
        "TARGET": np.random.binomial(1, 0.08, size=50000),
        "EXT_SOURCE_1": np.random.uniform(0, 1, size=50000),
        "EXT_SOURCE_2": np.random.uniform(0, 1, size=50000),
        "EXT_SOURCE_3": np.random.uniform(0, 1, size=50000),
        "AMT_INCOME_TOTAL": np.random.lognormal(11, 0.5, size=50000),
        "AMT_CREDIT": np.random.lognormal(12, 0.5, size=50000),
        "DAYS_BIRTH": np.random.randint(-25000, -6000, size=50000),
        "DAYS_EMPLOYED": np.random.randint(-10000, 0, size=50000),
        "FLAG_DOCUMENT_3": np.random.randint(0, 2, size=50000),
        "FLAG_DOCUMENT_6": np.random.randint(0, 2, size=50000),
        "NAME_CONTRACT_TYPE": np.random.choice(["Cash loans", "Revolving loans"], size=50000),
        "CODE_GENDER": np.random.choice(["M", "F"], size=50000),
        "NAME_INCOME_TYPE": np.random.choice(["Working", "Commercial associate", "Pensioner"], size=50000),
    })

    con.execute("CREATE OR REPLACE TABLE previous_application AS SELECT * FROM df")
    con.execute("CREATE OR REPLACE TABLE application_train AS SELECT * FROM app_df")
    print("Synthetic data generated: 50K previous applications, 50K current applications")
    return True


def build_funnel_stages(con):
    con.execute("""
        CREATE OR REPLACE TABLE funnel_stages AS
        SELECT
            SK_ID_PREV,
            SK_ID_CURR,
            NAME_CONTRACT_STATUS,
            CODE_REJECT_REASON,
            DAYS_DECISION,
            HOUR_APPR_PROCESS_START,
            NAME_CLIENT_TYPE,
            AMT_APPLICATION,
            RATE_DOWN_PAYMENT,
            FLAG_LAST_APPL_PER_CONTRACT,
            CASE
                WHEN NAME_CONTRACT_STATUS = 'Approved' THEN 5
                WHEN NAME_CONTRACT_STATUS = 'Unused offer' THEN 4
                WHEN NAME_CONTRACT_STATUS = 'Refused'
                     AND CODE_REJECT_REASON NOT IN ('HC', 'SCO') THEN 3
                WHEN NAME_CONTRACT_STATUS = 'Refused'
                     AND CODE_REJECT_REASON IN ('HC', 'SCO') THEN 2
                WHEN NAME_CONTRACT_STATUS = 'Canceled' THEN 1
                ELSE 1
            END AS final_stage,
            CASE
                WHEN NAME_CONTRACT_STATUS = 'Approved' THEN 'offer_acceptance'
                WHEN NAME_CONTRACT_STATUS = 'Unused offer' THEN 'offer_presentation'
                WHEN NAME_CONTRACT_STATUS = 'Refused'
                     AND CODE_REJECT_REASON NOT IN ('HC', 'SCO') THEN 'credit_check'
                WHEN NAME_CONTRACT_STATUS = 'Refused'
                     AND CODE_REJECT_REASON IN ('HC', 'SCO') THEN 'identity_verification'
                WHEN NAME_CONTRACT_STATUS = 'Canceled' THEN 'document_submission'
                ELSE 'document_submission'
            END AS drop_stage,
            ABS(DAYS_DECISION) AS days_since_decision,
            CASE WHEN FLAG_LAST_APPL_PER_CONTRACT = 'Y' THEN 'new_policy' ELSE 'old_policy' END AS policy_group
        FROM previous_application
        WHERE AMT_APPLICATION > 0
    """)
    print("Funnel stages built")


def compute_stage_metrics(con):
    metrics = con.execute("""
        WITH stage_counts AS (
            SELECT
                drop_stage,
                final_stage,
                COUNT(*) AS n_applicants,
                AVG(days_since_decision) AS avg_days_to_decision,
                AVG(AMT_APPLICATION) AS avg_loan_amount,
                COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () AS pct_total
            FROM funnel_stages
            GROUP BY drop_stage, final_stage
        )
        SELECT * FROM stage_counts
        ORDER BY final_stage DESC
    """).df()

    total = len(con.execute("SELECT * FROM funnel_stages").df())
    reached_stage_5 = len(con.execute(
        "SELECT * FROM funnel_stages WHERE final_stage = 5").df())
    conversion_rate = reached_stage_5 / total * 100

    print(f"\n=== Funnel Metrics ===")
    print(f"Total applications: {total:,}")
    print(f"End-to-end conversion rate: {conversion_rate:.2f}%")
    print(f"\nStage drop-off breakdown:")
    print(metrics[["drop_stage", "n_applicants", "avg_days_to_decision", "pct_total"]].to_string(index=False))

    metrics.to_csv("data/outputs/funnel_metrics.csv", index=False)
    return metrics, {"total": total, "conversion_rate": round(conversion_rate, 2)}


def run_module1():
    Path("data/outputs").mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(DB_PATH)
    load_raw_data(con)
    build_funnel_stages(con)
    metrics, summary = compute_stage_metrics(con)
    con.close()
    return metrics, summary
