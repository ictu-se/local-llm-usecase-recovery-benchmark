import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "data" / "tasks"


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def split_paths(value):
    text = str(value or "").strip()
    if not text:
        return []
    return [part.strip() for part in text.split("|") if part.strip()]


def build_prompt(task):
    original = task["prompt"]
    code_paths = task.get("code_paths_used", [])
    api_paths = split_paths(task.get("api_paths_gold", ""))
    code_list = "\n".join(f"- {path}" for path in code_paths) or "- none"
    api_list = "\n".join(f"- {path}" for path in api_paths) or "- none"

    code_marker = "Code evidence:"
    code_context = original.split(code_marker, 1)[1].strip() if code_marker in original else original

    return f"""You are analyzing code evidence to recover exactly one user-facing use case and its traceability paths.

Return ONLY valid JSON with these keys:
- actor: string
- use_case_name: string
- description: string
- main_flow: array of 3-6 short strings
- code_paths: array of strings selected only from Candidate code paths
- api_paths: array of strings selected only from Candidate API paths
- confidence: number between 0 and 1

Path selection rules:
- Select code_paths that directly implement the recovered use case.
- Select api_paths that belong to the recovered use case workflow.
- Do not invent paths.
- If all candidate paths are needed, return all of them.
- If a candidate path is only incidental or unrelated, omit it.
- Keep path strings exactly as shown in the candidate lists.

Naming rules:
- The API paths define the target workflow scope.
- Use "Manage <entity plural>" only when the API paths show several CRUD actions for the same entity.
- For a single dominant API verb such as create, view, collect, return, promote, apply, approve, upload, or export, preserve that verb in the use case name.

Repository: {task["repository_folder"]}
Actor hint: {task["actor_gold"]}

Candidate code paths:
{code_list}

Candidate API paths:
{api_list}

Code evidence:

{code_context}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(TASKS / "use_case_tasks_user_goal_v4.jsonl"))
    parser.add_argument("--output", default=str(TASKS / "use_case_tasks_user_goal_v4_path_fidelity.jsonl"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for task in load_jsonl(in_path):
            if args.limit and count >= args.limit:
                break
            new_task = dict(task)
            new_task["task_id"] = f'{task["task_id"]}_path'
            new_task["prompt_version"] = "user_goal_v4_path_fidelity"
            new_task["prompt"] = build_prompt(task)
            f.write(json.dumps(new_task, ensure_ascii=False) + "\n")
            count += 1

    print(json.dumps({"tasks_written": count, "output": str(out_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
