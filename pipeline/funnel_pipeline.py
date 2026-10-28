import duckdb
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

    if not (prev_app_path.exists() and app_train_path.exists()):
        raise FileNotFoundError(f"Home Credit files missing in {RAW_DIR}: need previous_application.csv and application_train.csv")

    con.execute(f"""
        CREATE OR REPLACE TABLE previous_application AS
        SELECT * FROM read_csv_auto('{prev_app_path}')
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE application_train AS
        SELECT * FROM read_csv_auto('{app_train_path}')
    """)
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
    summary["data_source"] = "home_credit_real"
    summary["rows"] = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ["previous_application", "application_train", "funnel_stages"]}
    con.close()
    return metrics, summary
