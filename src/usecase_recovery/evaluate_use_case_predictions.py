import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "results" / "runs"


def rel(path):
    return str(path.relative_to(ROOT))


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
    text = response.strip()
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5-coder:7b")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--predictions", default="")
    args = parser.parse_args()

    safe_model = args.model.replace(":", "_").replace("/", "_")
    safe_run = args.run_name.strip().replace(":", "_").replace("/", "_")
    run_dir = RUNS / (f"{safe_model}__{safe_run}" if safe_run else safe_model)
    pred_path = Path(args.predictions) if args.predictions else run_dir / "predictions.jsonl"
    metrics_path = run_dir / "metrics.csv"
    summary_path = run_dir / "summary.json"

    rows = []
    for rec in load_jsonl(pred_path):
        parsed, parse_error = parse_response(rec.get("response", ""))
        pred_name = parsed.get("use_case_name", "")
        pred_desc = parsed.get("description", "")
        gold_name = rec["gold"]["use_case_name"]
        gold_desc = rec["gold"]["description"]
        norm_pred = normalize(pred_name)
        norm_gold = normalize(gold_name)
        rows.append(
            {
                "task_id": rec["task_id"],
                "repository_folder": rec["repository_folder"],
                "model": rec["model"],
                "run_name": rec.get("run_name", ""),
                "prompt_version": rec.get("prompt_version", ""),
                "exact_name_match": int(norm_pred == norm_gold and bool(norm_gold)),
                "substring_name_match": int(bool(norm_pred) and (norm_pred in norm_gold or norm_gold in norm_pred)),
                "name_token_f1": round(token_f1(pred_name, gold_name), 4),
                "description_token_f1": round(token_f1(pred_desc, gold_desc), 4),
                "parse_error": parse_error,
                "runtime_error": rec.get("error", ""),
                "pred_use_case_name": pred_name,
                "gold_use_case_name": gold_name,
            }
        )

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    def avg(key):
        return round(sum(float(r[key]) for r in rows) / len(rows), 4) if rows else 0.0

    summary = {
        "model": args.model,
        "run_name": args.run_name,
        "n": len(rows),
        "exact_name_match": avg("exact_name_match"),
        "substring_name_match": avg("substring_name_match"),
        "name_token_f1": avg("name_token_f1"),
        "description_token_f1": avg("description_token_f1"),
        "parse_error_count": sum(1 for r in rows if r["parse_error"]),
        "runtime_error_count": sum(1 for r in rows if r["runtime_error"]),
        "metrics_csv": rel(metrics_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
