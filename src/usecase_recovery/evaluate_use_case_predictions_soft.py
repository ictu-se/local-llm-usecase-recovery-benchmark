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


SYNONYMS = {
    "signin": "login",
    "sign": "login",
    "authenticate": "login",
    "authentication": "login",
    "promo": "promocode",
    "promotion": "promote",
    "promotions": "promote",
    "allocated": "allocate",
    "allocation": "allocate",
    "allocations": "allocate",
    "registration": "register",
    "returned": "return",
    "collected": "collect",
    "collection": "collect",
    "created": "create",
    "creation": "create",
    "added": "add",
    "saved": "save",
    "saving": "save",
    "updated": "update",
    "updating": "update",
    "deleted": "delete",
    "deletion": "delete",
    "detail": "details",
    "requests": "applications",
    "request": "application",
    "feedbacks": "feedback",
    "marksheet": "marks",
    "tabulation": "sheet",
    "timetable": "schedule",
    "passwordless": "password",
}

STOP = {
    "a",
    "an",
    "and",
    "as",
    "for",
    "my",
    "new",
    "of",
    "or",
    "the",
    "to",
}


def safe_name(text):
    return text.strip().replace(":", "_").replace("/", "_")


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_response(response):
    text = response.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def canonical_token(token):
    token = token.lower()
    if token in SYNONYMS:
        token = SYNONYMS[token]
    if len(token) > 4 and token.endswith("ies"):
        token = token[:-3] + "y"
    elif len(token) > 4 and token.endswith("es"):
        token = token[:-2]
    elif len(token) > 3 and token.endswith("s") and token not in {"class", "status"}:
        token = token[:-1]
    return SYNONYMS.get(token, token)


def tokens(text):
    raw = re.findall(r"[a-zA-Z0-9]+", str(text or "").lower())
    return [canonical_token(t) for t in raw if canonical_token(t) not in STOP]


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


def soft_match(pred, gold):
    f1 = token_f1(pred, gold)
    p = set(tokens(pred))
    g = set(tokens(gold))
    if f1 >= 0.75:
        return 1
    if not p or not g:
        return 0
    # Good enough when the central verb and entity overlap.
    return int(len(p & g) >= 2 and f1 >= 0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5-coder:14b")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--predictions", default="")
    args = parser.parse_args()

    safe_model = safe_name(args.model)
    safe_run = safe_name(args.run_name)
    run_dir = RUNS / (f"{safe_model}__{safe_run}" if safe_run else safe_model)
    pred_path = Path(args.predictions) if args.predictions else run_dir / "predictions.jsonl"
    out_path = run_dir / "metrics_soft.csv"
    summary_path = run_dir / "summary_soft.json"

    rows = []
    for rec in load_jsonl(pred_path):
        parsed = parse_response(rec.get("response", ""))
        pred_name = parsed.get("use_case_name", "")
        gold_name = rec["gold"]["use_case_name"]
        f1 = token_f1(pred_name, gold_name)
        rows.append(
            {
                "task_id": rec["task_id"],
                "repository_folder": rec["repository_folder"],
                "soft_name_match": soft_match(pred_name, gold_name),
                "soft_name_token_f1": round(f1, 4),
                "pred_use_case_name": pred_name,
                "gold_use_case_name": gold_name,
            }
        )

    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "model": args.model,
        "run_name": args.run_name,
        "n": len(rows),
        "soft_name_match": round(sum(r["soft_name_match"] for r in rows) / len(rows), 4),
        "soft_name_token_f1": round(sum(float(r["soft_name_token_f1"]) for r in rows) / len(rows), 4),
        "metrics_soft_csv": rel(out_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
