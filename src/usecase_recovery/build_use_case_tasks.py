import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ANNOTATIONS = ROOT / "data" / "annotations"
REPOS = ROOT / "data" / "candidate_repos"
OUT_DIR = ROOT / "data" / "tasks"


def split_paths(value):
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [part.strip() for part in text.split("|") if part.strip()]


def read_text_sample(path, max_chars):
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"[READ_ERROR] {exc}"
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + "\n\n...[TRUNCATED]...\n\n" + text[-tail:]


def build_prompt(row, snippets, prompt_version):
    code_blocks = []
    for item in snippets:
        code_blocks.append(
            f"FILE: {item['path']}\n"
            f"```\n{item['content']}\n```"
        )
    code_context = "\n\n".join(code_blocks)
    api_hint = row.get("api_paths", "")
    test_hint = row.get("test_paths", "")
    actor_hint = row.get("actor", "")

    if prompt_version == "user_goal_v4":
        return f"""You are analyzing a software repository to recover exactly one user-facing use case from code evidence.

Return ONLY valid JSON with these keys:
- actor
- use_case_name
- description
- main_flow
- confidence

Most important rule:
- The API path hints define the target workflow scope. Code files may contain many actions; do not name the whole controller if the API hints point to only one action.

API-scope naming rules:
- If the API hints are only add/create/register/save POST endpoints, use "Create/Add/Register/Save <entity>", not "Manage <entity>".
- If the API hints are only GET detail endpoints with an id, use "View <entity> detail".
- If the API hints are only GET list/search endpoints, use "List/Search/View <entity plural>".
- If the API hints include GET edit/open form plus POST save, use "Open <entity> create or edit form" or "Save <entity>" according to the dominant API verb.
- If the API hints include collect/return/issue/approve/reply/verify/generate/promote/allocate/drop/enroll/start/stop/harvest/upload/download/export, keep that verb in the use case name.
- Use "Manage <entity plural>" only when API hints show several CRUD actions for the same entity, such as add/create + list/manage + edit/update + delete.
- If API hints are broad or absent, infer from tests first, then from code.

General naming:
- Prefer concise names with 2-7 words.
- Avoid implementation terms like controller, endpoint, database, mapper, serializer, form class.
- Preserve the actor hint when consistent with the code.
- main_flow must be a short list of 3-6 user/system steps.

Examples:
- API `/add | /add-book` -> "Create book"
- API `POST /stu` -> "Create student"
- API `GET /api/books/{{id}}` -> "View book detail"
- API `PUT /admin/orders/{{id}}/collect` -> "Mark order as collected"
- API `POST /students/promote...` -> "Promote students"
- API add + list + edit + delete staff endpoints -> "Manage staff"

Repository: {row['repository_folder']}
Actor hint: {actor_hint}
API path hints: {api_hint}
Test path hints: {test_hint}

Code evidence:

{code_context}
"""

    if prompt_version == "user_goal_v3":
        return f"""You are analyzing a software repository to recover exactly one user-facing use case from code evidence.

Return ONLY valid JSON with these keys:
- actor
- use_case_name
- description
- main_flow
- confidence

Naming policy:
- Match the functional scope shown by the API path hints and code, not just one function name.
- Use "Manage <entity plural>" ONLY when the evidence shows several CRUD actions for the same entity, such as add/create + list/view/manage + edit/update + delete.
- Do NOT use "Manage ..." for a single-purpose workflow such as apply, view, take, submit, pay, export, upload, download, search, authenticate, start, stop, issue, return, enroll, drop, or approve.
- If API hints include one dominant verb, keep that verb in the use case name: "Apply for leave", "View attendance", "Take quiz", "Submit homework answer", "Upload document file".
- If API hints include search/filter plus update/delete for the same entity, use a combined name such as "Search and update books" or "Filter overdue orders".
- Prefer concise names in Title Case or sentence case with 2-6 words. Avoid adding implementation terms like controller, endpoint, database, mapper, form, serializer.
- Use the Actor hint if it is consistent with the code.
- main_flow must be a short list of 3-6 user/system steps.

Examples of abstraction:
- add staff + manage staff + edit staff + delete staff -> "Manage staff"
- add result + edit result only -> "Add or update student result"
- student_apply_leave endpoint -> "Apply for leave"
- view_attendance endpoint -> "View attendance"
- search book + update book endpoints -> "Search and update books"

Repository: {row['repository_folder']}
Actor hint: {actor_hint}
API path hints: {api_hint}
Test path hints: {test_hint}

Code evidence:

{code_context}
"""

    if prompt_version == "user_goal_v2":
        return f"""You are analyzing a software repository to recover a USER-GOAL LEVEL use case from code evidence.

Return ONLY valid JSON with these keys:
- actor
- use_case_name
- description
- main_flow
- confidence

Critical abstraction rules:
- Name the use case at user-goal level, not at single-action level.
- If the evidence contains create/add, list/view, edit/update, delete/remove for the same entity, merge them into "Manage <entity plural>".
- If the evidence contains search/filter plus view/update for the same entity, use a broader management or lookup use case.
- Do not output narrow names such as "Add Staff", "Create Course", "List Books", "Update Student" when the code supports a broader workflow.
- Prefer names like "Manage staff", "Manage students", "Manage courses", "Search and update books", "Take attendance", "View results".
- Use the Actor hint if it is consistent with the code.
- Use API path hints and Test path hints as evidence of the workflow scope.
- Do not invent APIs or files that are not visible in the evidence.
- main_flow must be a short list of 3-6 user/system steps covering the broader workflow.

Repository: {row['repository_folder']}
Actor hint: {actor_hint}
API path hints: {api_hint}
Test path hints: {test_hint}

Code evidence:

{code_context}
"""

    return f"""You are analyzing a software repository to recover user-facing use cases from code evidence.

Return ONLY valid JSON with these keys:
- actor
- use_case_name
- description
- main_flow
- confidence

Rules:
- Infer one concise user-facing use case from the provided files.
- Use Vietnamese or English only if the evidence requires it; otherwise answer in English.
- Do not invent APIs or files that are not visible in the evidence.
- main_flow must be a short list of 3-6 steps.

Repository: {row['repository_folder']}
Actor hint: {actor_hint}
API path hints: {api_hint}
Test path hints: {test_hint}

Code evidence:

{code_context}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(ANNOTATIONS / "use_case_ground_truth_v1.csv"))
    parser.add_argument("--output", default=str(OUT_DIR / "use_case_tasks.jsonl"))
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=4)
    parser.add_argument("--max-chars-per-file", type=int, default=6000)
    parser.add_argument("--prompt-version", choices=["baseline_v1", "user_goal_v2", "user_goal_v3", "user_goal_v4"], default="baseline_v1")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    input_csv = Path(args.csv)
    output = Path(args.output)

    written = 0
    missing_files = 0
    with input_csv.open("r", encoding="utf-8-sig", newline="") as f_in, output.open(
        "w", encoding="utf-8", newline="\n"
    ) as f_out:
        reader = csv.DictReader(f_in)
        for idx, row in enumerate(reader):
            if args.max_rows and written >= args.max_rows:
                break
            repo_dir = REPOS / row["repository_folder"]
            snippets = []
            for rel in split_paths(row.get("code_paths"))[: args.max_files]:
                path = repo_dir / rel
                if not path.exists() or not path.is_file():
                    missing_files += 1
                    continue
                snippets.append(
                    {
                        "path": rel,
                        "content": read_text_sample(path, args.max_chars_per_file),
                    }
                )
            if not snippets:
                continue
            task = {
                "task_id": f"usecase_v1_{idx:04d}",
                "repository_folder": row["repository_folder"],
                "actor_gold": row["actor"],
                "use_case_name_gold": row["use_case_name"],
                "description_gold": row["description"],
                "main_flow_gold": row["main_flow"],
                "api_paths_gold": row.get("api_paths", ""),
                "test_paths_gold": row.get("test_paths", ""),
                "prompt_version": args.prompt_version,
                "prompt": build_prompt(row, snippets, args.prompt_version),
                "code_paths_used": [item["path"] for item in snippets],
            }
            f_out.write(json.dumps(task, ensure_ascii=False) + "\n")
            written += 1

    print(json.dumps({"tasks_written": written, "missing_code_files": missing_files, "prompt_version": args.prompt_version, "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
