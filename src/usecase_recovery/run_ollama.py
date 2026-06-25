import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "data" / "tasks" / "use_case_tasks_user_goal_v4.jsonl"
RUNS = ROOT / "results" / "runs"


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def call_ollama(model, prompt, temperature, num_ctx, timeout):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5-coder:7b")
    parser.add_argument("--tasks", default=str(TASKS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-name", default="")
    args = parser.parse_args()

    safe_model = args.model.replace(":", "_").replace("/", "_")
    safe_run = args.run_name.strip().replace(":", "_").replace("/", "_")
    out_dir = RUNS / (f"{safe_model}__{safe_run}" if safe_run else safe_model)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predictions.jsonl"

    done = set()
    if args.resume and out_path.exists():
        for row in load_jsonl(out_path):
            done.add(row["task_id"])

    mode = "a" if args.resume else "w"
    count = 0
    with out_path.open(mode, encoding="utf-8", newline="\n") as f_out:
        for task in load_jsonl(Path(args.tasks)):
            if args.limit and count >= args.limit:
                break
            if task["task_id"] in done:
                continue
            started = time.time()
            try:
                result = call_ollama(args.model, task["prompt"], args.temperature, args.num_ctx, args.timeout)
                response = result.get("response", "")
                error = ""
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                response = ""
                error = str(exc)
            elapsed = round(time.time() - started, 3)
            record = {
                "task_id": task["task_id"],
                "repository_folder": task["repository_folder"],
                "model": args.model,
                "run_name": args.run_name,
                "prompt_version": task.get("prompt_version", ""),
                "elapsed_seconds": elapsed,
                "response": response,
                "error": error,
                "gold": {
                    "actor": task["actor_gold"],
                    "use_case_name": task["use_case_name_gold"],
                    "description": task["description_gold"],
                    "main_flow": task["main_flow_gold"],
                    "api_paths": task.get("api_paths_gold", ""),
                    "test_paths": task.get("test_paths_gold", ""),
                },
                "code_paths_used": task["code_paths_used"],
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()
            count += 1
            print(f"{count}: {task['task_id']} {elapsed}s {error or 'ok'}")

    print(json.dumps({"written_or_attempted": count, "output": str(out_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
