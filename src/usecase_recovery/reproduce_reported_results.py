import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "results" / "runs"
AGG = ROOT / "results" / "aggregate"
SRC = ROOT / "src" / "usecase_recovery"


def safe_to_model(safe):
    if "_" not in safe:
        return safe
    family, size = safe.rsplit("_", 1)
    if size.endswith("b") or size in {"latest"}:
        return f"{family}:{size}"
    return safe


def run_python(script, *args):
    cmd = [sys.executable, str(SRC / script), *args]
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def split_run_dir(name):
    if "__" in name:
        model_safe, run_name = name.split("__", 1)
    else:
        model_safe, run_name = name, ""
    return safe_to_model(model_safe), run_name


def copy_if_exists(src, dst):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def rebuild_model_summary(scale_runs):
    rows = []
    for run_dir in scale_runs:
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8-sig"))
        soft_path = run_dir / "summary_soft.json"
        soft = json.loads(soft_path.read_text(encoding="utf-8-sig")) if soft_path.exists() else {}
        rows.append(
            {
                "model": summary.get("model", run_dir.name.split("__", 1)[0]),
                "run_name": summary.get("run_name", ""),
                "n": summary.get("n", 0),
                "exact_name_match": summary.get("exact_name_match", 0),
                "substring_name_match": summary.get("substring_name_match", 0),
                "name_token_f1": summary.get("name_token_f1", 0),
                "description_token_f1": summary.get("description_token_f1", 0),
                "soft_name_match": soft.get("soft_name_match", ""),
                "soft_name_token_f1": soft.get("soft_name_token_f1", ""),
                "parse_error_count": summary.get("parse_error_count", 0),
                "runtime_error_count": summary.get("runtime_error_count", 0),
            }
        )

    out = AGG / "model_scale_family_summary.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    AGG.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted(p for p in RUNS.iterdir() if p.is_dir() and (p / "predictions.jsonl").exists())
    scale_runs = []

    for run_dir in run_dirs:
        model, run_name = split_run_dir(run_dir.name)
        if run_name.endswith("_path_probe"):
            run_python("evaluate_path_fidelity.py", "--model", model, "--run-name", run_name)
        else:
            run_python("evaluate_use_case_predictions.py", "--model", model, "--run-name", run_name)
            run_python("evaluate_use_case_predictions_soft.py", "--model", model, "--run-name", run_name)
            if run_name == "user_goal_v4_scale_family":
                scale_runs.append(run_dir)

    run_python("analyze_extended_metrics.py", "--model", "qwen2.5-coder:14b", "--run-name", "user_goal_v4_probe")
    run_python("classify_error_taxonomy.py", "--model", "qwen2.5-coder:14b", "--run-name", "user_goal_v4_probe")
    run_python(
        "analyze_experimental_slices.py",
        "--model",
        "qwen2.5-coder:14b",
        "--run-name",
        "user_goal_v4_probe",
        "--path-model",
        "qwen2.5-coder:14b",
        "--path-run-name",
        "user_goal_v4_path_probe",
    )

    main_run = RUNS / "qwen2.5-coder_14b__user_goal_v4_probe"
    path_run = RUNS / "qwen2.5-coder_14b__user_goal_v4_path_probe"
    for name in [
        "metrics.csv",
        "metrics_soft.csv",
        "metrics_extended.csv",
        "metrics_by_repository.csv",
        "summary.json",
        "summary_soft.json",
        "summary_extended.json",
        "error_taxonomy_counts.csv",
        "error_taxonomy_details.csv",
        "experimental_slices_all.csv",
        "experimental_slices_by_framework.csv",
        "experimental_slices_by_domain.csv",
        "experimental_slices_by_code_path_count.csv",
        "experimental_slices_by_api_path_count.csv",
        "experimental_slices_by_test_evidence.csv",
        "confidence_calibration.csv",
        "path_fidelity_by_api_path_count.csv",
        "path_fidelity_by_code_path_count.csv",
        "experimental_slices_summary.json",
    ]:
        copy_if_exists(main_run / name, AGG / name)

    for name in ["metrics_path_fidelity.csv", "summary_path_fidelity.json"]:
        copy_if_exists(path_run / name, AGG / name)

    if scale_runs:
        rebuild_model_summary(scale_runs)

    print(json.dumps({"runs": len(run_dirs), "aggregate_dir": str(AGG)}, indent=2))


if __name__ == "__main__":
    main()
