import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "results" / "runs"


def safe_name(text):
    return text.strip().replace(":", "_").replace("/", "_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5-coder:7b")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    safe_model = safe_name(args.model)
    safe_run = safe_name(args.run_name)
    run_dir = RUNS / (f"{safe_model}__{safe_run}" if safe_run else safe_model)
    metrics_path = run_dir / "metrics.csv"
    out_path = run_dir / "ERROR_REVIEW_VI.md"

    with metrics_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    rows.sort(key=lambda r: float(r["name_token_f1"]))
    worst = rows[: args.top]
    exact = sum(int(r["exact_name_match"]) for r in rows)
    substr = sum(int(r["substring_name_match"]) for r in rows)
    avg_name = sum(float(r["name_token_f1"]) for r in rows) / len(rows)
    avg_desc = sum(float(r["description_token_f1"]) for r in rows) / len(rows)

    title = f"`{args.model}`" + (f" / `{args.run_name}`" if args.run_name else "")
    lines = [
        f"# Review lỗi baseline Ollama: {title}",
        "",
        "## Tóm tắt",
        "",
        f"- Số task: {len(rows)}",
        f"- Exact match tên use case: {exact}/{len(rows)} = {exact/len(rows):.3f}",
        f"- Substring match tên use case: {substr}/{len(rows)} = {substr/len(rows):.3f}",
        f"- Trung bình name token F1: {avg_name:.3f}",
        f"- Trung bình description token F1: {avg_desc:.3f}",
        "",
        "## Các lỗi tên use case tệ nhất",
        "",
        "| task_id | repo | gold | pred | name_f1 |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for r in worst:
        gold = r["gold_use_case_name"].replace("|", "/")
        pred = r["pred_use_case_name"].replace("|", "/")
        lines.append(
            f"| `{r['task_id']}` | `{r['repository_folder']}` | {gold} | {pred} | {float(r['name_token_f1']):.3f} |"
        )

    lines += [
        "",
        "## Nhận xét nhanh",
        "",
        "- Chạy ổn định nếu `parse_error_count` và `runtime_error_count` bằng 0 trong `summary.json`.",
        "- Nếu tên use case còn quá hẹp, cần tăng ràng buộc gom CRUD/list/search/update/delete thành user-goal cấp cao.",
        "- Nếu tên use case quá rộng, cần đưa thêm API/test hints hoặc giảm số file code gây nhiễu.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8-sig")
    print(json.dumps({"output": str(out_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
