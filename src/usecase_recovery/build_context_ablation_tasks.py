import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = ROOT / "data" / "tasks"


def strip_api_scope(prompt):
    prompt = re.sub(
        r"Most important rule:\n- The API path hints define.*?\n\n",
        "Most important rule:\n- Recover the target user-facing use case from the provided code and test evidence.\n\n",
        prompt,
        flags=re.S,
    )
    prompt = re.sub(
        r"API-scope naming rules:\n.*?\n\nGeneral naming:",
        "General naming:",
        prompt,
        flags=re.S,
    )
    prompt = re.sub(r"^API path hints:.*\n", "", prompt, flags=re.M)
    prompt = prompt.replace("If API hints are broad or absent, infer from tests first, then from code.\n", "")
    return prompt


def main():
    src = TASK_DIR / "use_case_tasks_user_goal_v4.jsonl"
    dst = TASK_DIR / "use_case_tasks_user_goal_v4_no_api.jsonl"
    count = 0
    with src.open("r", encoding="utf-8") as f_in, dst.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            rec = json.loads(line)
            rec["prompt_version"] = "user_goal_v4_no_api"
            rec["prompt"] = strip_api_scope(rec["prompt"])
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    print(f"Wrote {count} tasks to {dst}")


if __name__ == "__main__":
    main()
