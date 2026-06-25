import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "results" / "runs"


def rel(path):
    return str(path.relative_to(ROOT))


def safe_name(text):
    return text.strip().replace(":", "_").replace("/", "_")


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize(text):
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def tokens(text):
    return re.findall(r"[a-z0-9]+", normalize(text))


def token_f1(pred, gold):
    p = tokens(pred)
    g = tokens(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def parse_response(response):
    text = str(response or "").strip()
    if "```" in text:
        text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.I).strip()
        text = re.sub(r"```$", "", text.strip()).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data, ""
    except json.JSONDecodeError as exc:
        return {}, str(exc)
    return {}, "response is not a JSON object"


def mean(rows, key):
    return sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0


def bootstrap_ci(rows, key, iterations=5000, seed=42):
    rng = random.Random(seed)
    values = [float(row[key]) for row in rows]
    if not values:
        return 0.0, 0.0
    n = len(values)
    estimates = []
    for _ in range(iterations):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        estimates.append(sum(sample) / n)
    estimates.sort()
    lo = estimates[int(0.025 * iterations)]
    hi = estimates[int(0.975 * iterations)]
    return lo, hi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5-coder:14b")
    parser.add_argument("--run-name", default="user_goal_v4_probe")
    parser.add_argument("--predictions", default="")
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()

    run_dir = RUNS / f"{safe_name(args.model)}__{safe_name(args.run_name)}"
    pred_path = Path(args.predictions) if args.predictions else run_dir / "predictions.jsonl"

    rows = []
    for rec in load_jsonl(pred_path):
        parsed, parse_error = parse_response(rec.get("response", ""))
        runtime_error = rec.get("error", "")
        pred_actor = parsed.get("actor", "")
        pred_name = parsed.get("use_case_name", "")
        pred_desc = parsed.get("description", "")
        gold_actor = rec["gold"]["actor"]
        gold_name = rec["gold"]["use_case_name"]
        gold_desc = rec["gold"]["description"]
        valid_prediction = int(not parse_error and not runtime_error and bool(normalize(pred_name)))
        rows.append(
            {
                "task_id": rec["task_id"],
                "repository_folder": rec["repository_folder"],
                "actor_exact_match": int(normalize(pred_actor) == normalize(gold_actor) and bool(normalize(gold_actor))),
                "actor_token_f1": token_f1(pred_actor, gold_actor),
                "exact_name_match": int(normalize(pred_name) == normalize(gold_name) and bool(normalize(gold_name))),
                "substring_name_match": int(
                    bool(normalize(pred_name))
                    and (normalize(pred_name) in normalize(gold_name) or normalize(gold_name) in normalize(pred_name))
                ),
                "name_token_f1": token_f1(pred_name, gold_name),
                "description_token_f1": token_f1(pred_desc, gold_desc),
                "valid_prediction": valid_prediction,
                "parse_error": int(bool(parse_error)),
                "runtime_error": int(bool(runtime_error)),
                "gold_actor": gold_actor,
                "pred_actor": pred_actor,
                "gold_use_case_name": gold_name,
                "pred_use_case_name": pred_name,
            }
        )

    out_rows = run_dir / "metrics_extended.csv"
    with out_rows.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            for key in ["actor_token_f1", "name_token_f1", "description_token_f1"]:
                formatted[key] = round(float(formatted[key]), 4)
            writer.writerow(formatted)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["repository_folder"]].append(row)

    repo_rows = []
    for repo, items in sorted(grouped.items()):
        repo_rows.append(
            {
                "repository_folder": repo,
                "n": len(items),
                "actor_exact_match": round(mean(items, "actor_exact_match"), 4),
                "actor_token_f1": round(mean(items, "actor_token_f1"), 4),
                "exact_name_match": round(mean(items, "exact_name_match"), 4),
                "substring_name_match": round(mean(items, "substring_name_match"), 4),
                "name_token_f1": round(mean(items, "name_token_f1"), 4),
                "description_token_f1": round(mean(items, "description_token_f1"), 4),
                "valid_prediction": round(mean(items, "valid_prediction"), 4),
            }
        )

    out_repo = run_dir / "metrics_by_repository.csv"
    with out_repo.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = list(repo_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(repo_rows)

    ci_keys = [
        "actor_exact_match",
        "actor_token_f1",
        "exact_name_match",
        "substring_name_match",
        "name_token_f1",
        "description_token_f1",
        "valid_prediction",
    ]
    summary = {
        "model": args.model,
        "run_name": args.run_name,
        "n": len(rows),
        "repositories": len(grouped),
        "path_fidelity_status": "not_computed_predictions_do_not_include_paths",
        "omission_or_coverage_definition": "valid parsed prediction with non-empty use_case_name",
        "metrics_extended_csv": rel(out_rows),
        "metrics_by_repository_csv": rel(out_repo),
    }
    for key in ci_keys:
        lo, hi = bootstrap_ci(rows, key, iterations=args.bootstrap)
        summary[key] = round(mean(rows, key), 4)
        summary[f"{key}_ci95_low"] = round(lo, 4)
        summary[f"{key}_ci95_high"] = round(hi, 4)

    out_summary = run_dir / "summary_extended.json"
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
