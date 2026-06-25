import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "data" / "tasks" / "use_case_tasks_user_goal_v4.jsonl"
RUN = ROOT / "results" / "runs" / "qwen2.5-coder_14b__user_goal_v4_probe"
PATH_RUN = ROOT / "results" / "runs" / "qwen2.5-coder_14b__user_goal_v4_path_probe"


def safe_name(text):
    return text.strip().replace(":", "_").replace("/", "_")


FRAMEWORK = {
    "4jean__lav_sms": "Laravel/PHP",
    "abdulwahid880__School-Management-system-in-laravel-": "Laravel/PHP",
    "adityagoyal222__django-lms": "Django/Python",
    "AdityaKshettri__Student-Database-Management-System-using-JAVA-Spring-Boot": "Spring/Java",
    "Asitha123__Java-Spring-boot-Student-Management-System": "Spring/Java",
    "engripaye__library-management-system": "Spring/Java",
    "JacksonGao1999__StudentManagement": "JavaScript",
    "jobic10__student-management-using-django": "Django/Python",
    "knowledgefactory4u__librarymanagementsystem": "Spring/Java",
    "mehmetpekdemir__Library-Management-System": "Spring/Java",
    "mwinamijr__django-scms": "Django/Python",
    "openfun__marsha": "Django/Python",
    "RameshMF__library-management-system": "Spring/Java",
    "SkyCascade__SkyLearn": "TypeScript/Node",
    "thakurpdhiraj__library_management_system": "Microservices/Java",
    "tough-dev-school__education-backend": "TypeScript/Node",
}

DOMAIN = {
    "4jean__lav_sms": "School management",
    "abdulwahid880__School-Management-system-in-laravel-": "School management",
    "adityagoyal222__django-lms": "Learning management",
    "AdityaKshettri__Student-Database-Management-System-using-JAVA-Spring-Boot": "Student management",
    "Asitha123__Java-Spring-boot-Student-Management-System": "Student management",
    "engripaye__library-management-system": "Library management",
    "JacksonGao1999__StudentManagement": "Student management",
    "jobic10__student-management-using-django": "Student management",
    "knowledgefactory4u__librarymanagementsystem": "Library management",
    "mehmetpekdemir__Library-Management-System": "Library management",
    "mwinamijr__django-scms": "School management",
    "openfun__marsha": "Educational media",
    "RameshMF__library-management-system": "Library management",
    "SkyCascade__SkyLearn": "Learning management",
    "thakurpdhiraj__library_management_system": "Library management",
    "tough-dev-school__education-backend": "Learning management",
}


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def split_paths(value):
    return [p.strip().strip("/") for p in (value or "").split("|") if p.strip()]


def parse_response(text):
    text = (text or "").strip()
    if "```" in text:
        text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
        text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def load_predictions(path):
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            parsed = parse_response(rec.get("response", ""))
            out[rec["task_id"]] = parsed
    return out


def avg(rows, key):
    return sum(float(r[key]) for r in rows) / len(rows) if rows else 0.0


def summarize(rows, group_key, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[row[group_key]].append(row)
    summary = []
    for group, items in sorted(groups.items()):
        out = {group_key: group, "n": len(items)}
        for metric in metrics:
            out[metric] = round(avg(items, metric), 4)
        summary.append(out)
    return summary


def bin_count(n):
    if n <= 1:
        return "1"
    if n <= 3:
        return "2-3"
    return "4+"


def conf_bin(conf, missing=False, non_numeric=False):
    if missing:
        return "missing"
    if non_numeric:
        return "non-numeric"
    if conf < 0.5:
        return "<0.50"
    if conf < 0.7:
        return "0.50-0.69"
    if conf < 0.9:
        return "0.70-0.89"
    return "0.90-1.00"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5-coder:14b")
    parser.add_argument("--run-name", default="user_goal_v4_probe")
    parser.add_argument("--path-model", default="qwen2.5-coder:14b")
    parser.add_argument("--path-run-name", default="user_goal_v4_path_probe")
    parser.add_argument("--tasks", default=str(TASKS))
    args = parser.parse_args()

    run = ROOT / "results" / "runs" / f"{safe_name(args.model)}__{safe_name(args.run_name)}"
    path_run = ROOT / "results" / "runs" / f"{safe_name(args.path_model)}__{safe_name(args.path_run_name)}"

    tasks = read_jsonl(Path(args.tasks))
    metrics = {r["task_id"]: r for r in read_csv(run / "metrics_extended.csv")}
    path_metrics_raw = read_csv(path_run / "metrics_path_fidelity.csv")
    path_metrics = {r["task_id"].replace("_path", ""): r for r in path_metrics_raw}
    predictions = load_predictions(run / "predictions.jsonl")

    rows = []
    for task in tasks:
        task_id = task["task_id"]
        metric = metrics[task_id]
        repo = task["repository_folder"]
        parsed = predictions.get(task_id, {})
        code_count = len(task.get("code_paths_used", []))
        api_count = len(split_paths(task.get("api_paths_gold", "")))
        test_count = len(split_paths(task.get("test_paths_gold", "")))
        confidence = parsed.get("confidence", "")
        missing_confidence = confidence in ("", None)
        non_numeric_confidence = False
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
            non_numeric_confidence = not missing_confidence
        if confidence > 1.0:
            confidence = confidence / 100.0
        confidence = max(0.0, min(1.0, confidence))
        row = {
            "task_id": task_id,
            "repository_folder": repo,
            "framework": FRAMEWORK.get(repo, "Other"),
            "domain": DOMAIN.get(repo, "Other"),
            "code_path_count": code_count,
            "api_path_count": api_count,
            "test_path_count": test_count,
            "code_path_bin": bin_count(code_count),
            "api_path_bin": bin_count(api_count),
            "has_test_evidence": "yes" if test_count else "no",
            "confidence_bin": conf_bin(confidence, missing_confidence, non_numeric_confidence),
            "confidence": round(confidence, 4),
            "exact_name_match": float(metric["exact_name_match"]),
            "substring_name_match": float(metric["substring_name_match"]),
            "name_token_f1": float(metric["name_token_f1"]),
            "description_token_f1": float(metric["description_token_f1"]),
            "soft_success": 1.0 if float(metric["name_token_f1"]) >= 0.7 else 0.0,
        }
        if task_id in path_metrics:
            pm = path_metrics[task_id]
            row.update(
                {
                    "code_path_f1": float(pm["code_path_f1"]),
                    "api_path_f1": float(pm["api_path_f1"]),
                    "code_path_exact": float(pm["code_path_exact"]),
                    "api_path_exact": float(pm["api_path_exact"]),
                }
            )
        rows.append(row)

    out_dir = run
    write_csv(out_dir / "experimental_slices_all.csv", rows)
    main_metrics = ["exact_name_match", "substring_name_match", "name_token_f1", "description_token_f1"]
    write_csv(out_dir / "experimental_slices_by_framework.csv", summarize(rows, "framework", main_metrics))
    write_csv(out_dir / "experimental_slices_by_domain.csv", summarize(rows, "domain", main_metrics))
    write_csv(out_dir / "experimental_slices_by_code_path_count.csv", summarize(rows, "code_path_bin", main_metrics))
    write_csv(out_dir / "experimental_slices_by_api_path_count.csv", summarize(rows, "api_path_bin", main_metrics))
    write_csv(out_dir / "experimental_slices_by_test_evidence.csv", summarize(rows, "has_test_evidence", main_metrics))
    write_csv(out_dir / "confidence_calibration.csv", summarize(rows, "confidence_bin", ["confidence", "exact_name_match", "soft_success", "name_token_f1"]))
    write_csv(out_dir / "path_fidelity_by_api_path_count.csv", summarize(rows, "api_path_bin", ["code_path_f1", "api_path_f1", "code_path_exact", "api_path_exact"]))
    write_csv(out_dir / "path_fidelity_by_code_path_count.csv", summarize(rows, "code_path_bin", ["code_path_f1", "api_path_f1", "code_path_exact", "api_path_exact"]))

    summary = {
        "n": len(rows),
        "frameworks": len({r["framework"] for r in rows}),
        "domains": len({r["domain"] for r in rows}),
        "outputs": [
            "experimental_slices_by_framework.csv",
            "experimental_slices_by_domain.csv",
            "experimental_slices_by_code_path_count.csv",
            "experimental_slices_by_api_path_count.csv",
            "experimental_slices_by_test_evidence.csv",
            "confidence_calibration.csv",
            "path_fidelity_by_api_path_count.csv",
            "path_fidelity_by_code_path_count.csv",
        ],
    }
    (out_dir / "experimental_slices_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
