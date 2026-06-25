import argparse
import csv
import json
import re
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
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def split_gold_paths(value):
    text = str(value or "").strip()
    if not text:
        return []
    return [part.strip() for part in text.split("|") if part.strip()]


def normalize_path(path):
    return str(path or "").strip().replace("\\", "/").strip("/")


def as_path_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [normalize_path(v) for v in value if normalize_path(v)]
    if isinstance(value, str):
        if "|" in value:
            return [normalize_path(v) for v in value.split("|") if normalize_path(v)]
        if "," in value and "\n" not in value:
            return [normalize_path(v) for v in value.split(",") if normalize_path(v)]
        return [normalize_path(value)] if normalize_path(value) else []
    return []


def set_scores(pred, gold):
    p = {normalize_path(x) for x in pred if normalize_path(x)}
    g = {normalize_path(x) for x in gold if normalize_path(x)}
    if not p and not g:
        return 1.0, 1.0, 1.0, 1
    if not p or not g:
        return 0.0, 0.0, 0.0, 0
    overlap = len(p & g)
    precision = overlap / len(p)
    recall = overlap / len(g)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    exact = int(p == g)
    return precision, recall, f1, exact


def mean(rows, key):
    return sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5-coder:14b")
    parser.add_argument("--run-name", default="user_goal_v4_path_probe")
    parser.add_argument("--predictions", default="")
    args = parser.parse_args()

    run_dir = RUNS / f"{safe_name(args.model)}__{safe_name(args.run_name)}"
    pred_path = Path(args.predictions) if args.predictions else run_dir / "predictions.jsonl"

    rows = []
    for rec in load_jsonl(pred_path):
        parsed = parse_response(rec.get("response", ""))
        pred_code = as_path_list(parsed.get("code_paths"))
        pred_api = as_path_list(parsed.get("api_paths"))
        gold_code = [normalize_path(x) for x in rec.get("code_paths_used", [])]
        gold_api = [normalize_path(x) for x in split_gold_paths(rec.get("gold", {}).get("api_paths", ""))]
        cp, cr, cf, ce = set_scores(pred_code, gold_code)
        ap, ar, af, ae = set_scores(pred_api, gold_api)
        rows.append(
            {
                "task_id": rec["task_id"],
                "repository_folder": rec["repository_folder"],
                "code_path_precision": round(cp, 4),
                "code_path_recall": round(cr, 4),
                "code_path_f1": round(cf, 4),
                "code_path_exact": ce,
                "api_path_precision": round(ap, 4),
                "api_path_recall": round(ar, 4),
                "api_path_f1": round(af, 4),
                "api_path_exact": ae,
                "pred_code_paths": " | ".join(pred_code),
                "gold_code_paths": " | ".join(gold_code),
                "pred_api_paths": " | ".join(pred_api),
                "gold_api_paths": " | ".join(gold_api),
                "parse_ok": int(bool(parsed)),
                "runtime_error": int(bool(rec.get("error", ""))),
            }
        )

    out_csv = run_dir / "metrics_path_fidelity.csv"
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "model": args.model,
        "run_name": args.run_name,
        "n": len(rows),
        "code_path_precision": round(mean(rows, "code_path_precision"), 4),
        "code_path_recall": round(mean(rows, "code_path_recall"), 4),
        "code_path_f1": round(mean(rows, "code_path_f1"), 4),
        "code_path_exact": round(mean(rows, "code_path_exact"), 4),
        "api_path_precision": round(mean(rows, "api_path_precision"), 4),
        "api_path_recall": round(mean(rows, "api_path_recall"), 4),
        "api_path_f1": round(mean(rows, "api_path_f1"), 4),
        "api_path_exact": round(mean(rows, "api_path_exact"), 4),
        "parse_ok": round(mean(rows, "parse_ok"), 4),
        "runtime_error_count": sum(int(r["runtime_error"]) for r in rows),
        "metrics_path_fidelity_csv": rel(out_csv),
    }
    out_summary = run_dir / "summary_path_fidelity.json"
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
