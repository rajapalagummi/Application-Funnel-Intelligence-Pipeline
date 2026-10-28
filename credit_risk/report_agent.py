import json
import re
import sys
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import TypedDict
from langgraph.graph import StateGraph, END

OUTPUTS = Path("data/outputs")
REPORTS = Path("reports")
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"
MAX_RETRIES = 2
NUMBER = re.compile(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?")


class State(TypedDict):
    facts: dict
    sources: list
    draft: str
    unsupported: list
    attempts: int
    numbers_checked: int
    status: str


def flatten(obj):
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, dict):
        return [x for v in obj.values() for x in flatten(v)]
    if isinstance(obj, list):
        return [x for v in obj for x in flatten(v)]
    return []


def numbers_in(text):
    out = []
    for token in NUMBER.findall(text):
        value = float(token.replace(",", ""))
        if "." not in token and (abs(value) <= 10 or 1900 <= value <= 2100):
            continue
        out.append(token)
    return out


def supported(token, sources):
    value = float(token.replace(",", ""))
    decimals = len(token.split(".")[1]) if "." in token else 0
    tol = 0.5 * 10 ** -decimals + 1e-9
    return any(abs(c - value) <= tol for s in sources for c in (s, s * 100))


def reader(state):
    validation = json.load(open(OUTPUTS / "validation_results.json"))
    metrics = json.load(open(OUTPUTS / "credit_model_metrics.json"))
    drift = pd.read_csv(OUTPUTS / "drift_monitoring.csv")
    facts = {
        "scorecard_test": validation["discrimination"]["scorecard"],
        "challenger_test": validation["discrimination"]["challenger"],
        "benchmark": validation["benchmark"],
        "calibration": {k: v for k, v in validation["calibration"].items() if k != "deciles"},
        "stability": validation["stability"],
        "validation_summary": validation["summary"],
        "features_in_model": metrics["features"]["n_final"],
        "funnel_drift_monitoring": drift.to_dict(orient="records"),
    }
    sources = flatten(validation) + flatten(metrics) + flatten(drift.to_dict(orient="records"))
    return {"facts": facts, "sources": sources, "attempts": 0, "unsupported": []}


def writer(state):
    prompt = (
        "You are a model risk analyst. Write a monthly model monitoring summary in Markdown for a credit PD scorecard, "
        "with sections: Overview, Discrimination, Calibration, Stability, Validation status, Actions. "
        "Use ONLY numbers that appear in the JSON facts below, copied exactly; do not compute new numbers or round differently. "
        "Keep it under 300 words.\n\nFACTS:\n" + json.dumps(state["facts"], indent=1)
    )
    if state["unsupported"]:
        prompt += "\n\nYour previous draft used numbers not present in the facts: " + ", ".join(state["unsupported"]) + ". Remove or correct them."
    r = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0}}, timeout=900)
    return {"draft": r.json()["response"], "attempts": state["attempts"] + 1}


def validator(state):
    tokens = numbers_in(state["draft"])
    bad = [t for t in tokens if not supported(t, state["sources"])]
    if not bad:
        result = "approved"
    elif state["attempts"] <= MAX_RETRIES:
        result = "retry"
    else:
        result = "human_review"
    return {"unsupported": bad, "numbers_checked": len(tokens), "status": result}


def build():
    g = StateGraph(State)
    g.add_node("reader", reader)
    g.add_node("writer", writer)
    g.add_node("validator", validator)
    g.set_entry_point("reader")
    g.add_edge("reader", "writer")
    g.add_edge("writer", "validator")
    g.add_conditional_edges("validator", lambda s: "writer" if s["status"] == "retry" else END)
    return g.compile()


def log_run(state):
    path = OUTPUTS / "agent_runs.json"
    log = json.load(open(path)) if path.exists() else {"runs": []}
    checked = state["numbers_checked"]
    log["runs"].append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": MODEL,
        "attempts": state["attempts"],
        "retries": state["attempts"] - 1,
        "numbers_checked": checked,
        "numbers_supported": checked - len(state["unsupported"]),
        "status": state["status"],
        "unsupported": state["unsupported"],
    })
    runs = log["runs"]
    total_checked = sum(r["numbers_checked"] for r in runs)
    total_supported = sum(r["numbers_supported"] for r in runs)
    log["summary"] = {
        "total_runs": len(runs),
        "approved": sum(r["status"] == "approved" for r in runs),
        "human_review": sum(r["status"] == "human_review" for r in runs),
        "approved_first_attempt": sum(r["status"] == "approved" and r["attempts"] == 1 for r in runs),
        "total_retries": sum(r["retries"] for r in runs),
        "numbers_checked_total": total_checked,
        "numbers_supported_total": total_supported,
        "guardrail_pass_rate": round(total_supported / total_checked, 4) if total_checked else 0.0,
    }
    with open(path, "w") as f:
        json.dump(log, f, indent=2)
    return log["summary"]


def run(n_runs=1):
    REPORTS.mkdir(exist_ok=True)
    app = build()
    for i in range(n_runs):
        state = app.invoke({})
        header = f"<!-- generated {datetime.now().isoformat(timespec='seconds')} | status: {state['status']} | attempts: {state['attempts']} | numbers checked: {state['numbers_checked']} -->\n"
        if state["status"] == "human_review":
            header += f"> FLAGGED FOR HUMAN REVIEW - unsupported numbers: {', '.join(state['unsupported'])}\n\n"
        (REPORTS / "monitoring_summary.md").write_text(header + state["draft"].strip() + "\n")
        summary = log_run(state)
        print(f"[agent] run {i + 1}/{n_runs}: {state['status']} after {state['attempts']} attempt(s), {state['numbers_checked']} numbers checked")
    print(f"[agent] {summary}")
    return summary


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
