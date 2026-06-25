import csv
import argparse
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results" / "runs" / "qwen2.5-coder_14b__user_goal_v4_probe"
INP = RUN / "metrics_extended.csv"
OUT = RUN / "error_taxonomy_counts.csv"
DETAIL = RUN / "error_taxonomy_details.csv"


def safe_name(text):
    return text.strip().replace(":", "_").replace("/", "_")

BROAD_WORDS = {"manage", "maintain", "administer", "handle"}
SPECIFIC_WORDS = {
    "add",
    "create",
    "view",
    "list",
    "search",
    "edit",
    "update",
    "delete",
    "approve",
    "reject",
    "apply",
    "submit",
    "upload",
    "download",
    "export",
    "import",
    "pay",
    "mark",
    "review",
    "return",
    "issue",
    "allocate",
    "promote",
    "open",
    "load",
    "refresh",
    "authenticate",
    "login",
    "sign",
    "stop",
    "harvest",
}


def first_token(text):
    text = (text or "").strip().lower().replace("/", " ")
    return text.split()[0] if text.split() else ""


def classify(row):
    exact = int(float(row["exact_name_match"]))
    substr = int(float(row["substring_name_match"]))
    f1 = float(row["name_token_f1"])
    gold_first = first_token(row["gold_use_case_name"])
    pred_first = first_token(row["pred_use_case_name"])

    if exact:
        return "Exact match"
    if substr or f1 >= 0.7:
        return "Near match / phrasing"
    if gold_first in BROAD_WORDS and pred_first in SPECIFIC_WORDS:
        return "Too narrow"
    if pred_first in BROAD_WORDS and gold_first in SPECIFIC_WORDS:
        return "Too broad"
    return "Semantic drift"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5-coder:14b")
    parser.add_argument("--run-name", default="user_goal_v4_probe")
    args = parser.parse_args()

    run = ROOT / "results" / "runs" / f"{safe_name(args.model)}__{safe_name(args.run_name)}"
    inp = run / "metrics_extended.csv"
    out = run / "error_taxonomy_counts.csv"
    detail = run / "error_taxonomy_details.csv"

    with inp.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    details = []
    counts = Counter()
    for row in rows:
        label = classify(row)
        counts[label] += 1
        details.append(
            {
                "task_id": row["task_id"],
                "repository_folder": row["repository_folder"],
                "gold_use_case_name": row["gold_use_case_name"],
                "pred_use_case_name": row["pred_use_case_name"],
                "name_token_f1": row["name_token_f1"],
                "category": label,
            }
        )

    ordered = ["Exact match", "Near match / phrasing", "Too narrow", "Too broad", "Semantic drift"]
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "count", "percent"])
        writer.writeheader()
        for category in ordered:
            writer.writerow(
                {
                    "category": category,
                    "count": counts[category],
                    "percent": round(counts[category] / len(rows), 4),
                }
            )
    with detail.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(details[0].keys()))
        writer.writeheader()
        writer.writerows(details)
    print(dict(counts))


if __name__ == "__main__":
    main()
