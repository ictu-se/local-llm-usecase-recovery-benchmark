import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = ROOT / "data" / "tasks"
SRC = TASK_DIR / "use_case_tasks_user_goal_v4.jsonl"


VARIANTS = {
    "evidence_code_only": {
        "use_api": False,
        "use_test": False,
        "use_code": True,
        "purpose": "recover the use case from source-code evidence only",
    },
    "evidence_api_only": {
        "use_api": True,
        "use_test": False,
        "use_code": False,
        "purpose": "recover the use case from API or route-path evidence only",
    },
    "evidence_code_api_no_test": {
        "use_api": True,
        "use_test": False,
        "use_code": True,
        "purpose": "recover the use case from source-code and API evidence, with test evidence withheld",
    },
    "evidence_code_test_no_api": {
        "use_api": False,
        "use_test": True,
        "use_code": True,
        "purpose": "recover the use case from source-code and test evidence, with API evidence withheld",
    },
    "evidence_code_api_test": {
        "use_api": True,
        "use_test": True,
        "use_code": True,
        "purpose": "recover the use case from the full source-code, API, and test evidence bundle",
    },
}


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def code_context_from_prompt(prompt):
    marker = "Code evidence:\n\n"
    if marker not in prompt:
        return "[Code evidence unavailable]"
    return prompt.split(marker, 1)[1].strip()


def build_prompt(task, variant_name, cfg):
    actor_hint = task.get("actor_gold", "")
    api_hint = task.get("api_paths_gold", "") if cfg["use_api"] else "[withheld]"
    test_hint = task.get("test_paths_gold", "") if cfg["use_test"] else "[withheld]"
    code_context = code_context_from_prompt(task["prompt"]) if cfg["use_code"] else "[source-code evidence withheld]"

    evidence_rules = []
    if cfg["use_api"]:
        evidence_rules.extend(
            [
                "- Treat API or route paths as the strongest signal for workflow scope.",
                "- If API paths show several CRUD actions for one entity, a management-level use case is appropriate.",
                "- If API paths show one dominant action, keep that action in the use case name.",
            ]
        )
    else:
        evidence_rules.append("- API or route-path hints are withheld; infer scope from the remaining evidence.")

    if cfg["use_test"]:
        evidence_rules.append("- Use test paths as supporting behavioral evidence.")
    else:
        evidence_rules.append("- Test evidence is withheld for this ablation condition.")

    if cfg["use_code"]:
        evidence_rules.append("- Use code names, route declarations, controllers, forms, models, and nearby logic as evidence.")
    else:
        evidence_rules.append("- Source-code snippets are withheld; do not invent files, controllers, or implementation details.")

    return f"""You are analyzing a software repository to recover exactly one user-facing use case.

This is an evidence ablation condition: {variant_name}.
Your task is to {cfg['purpose']}.

Return ONLY valid JSON with these keys:
- actor
- use_case_name
- description
- main_flow
- confidence

Evidence policy:
{chr(10).join(evidence_rules)}

Naming policy:
- Prefer concise names with 2-7 words.
- Name the actor's goal, not an implementation artifact.
- Avoid terms such as controller, endpoint, database, mapper, serializer, or form class.
- main_flow must be a short list of 3-6 user/system steps.
- If evidence is insufficient, provide the best evidence-grounded use case and lower the confidence.

Repository: {task['repository_folder']}
Actor hint: {actor_hint}
API path hints: {api_hint}
Test path hints: {test_hint}

Code evidence:

{code_context}
"""


def main():
    tasks = list(load_jsonl(SRC))
    for variant_name, cfg in VARIANTS.items():
        out = TASK_DIR / f"use_case_tasks_{variant_name}.jsonl"
        with out.open("w", encoding="utf-8", newline="\n") as f:
            for task in tasks:
                rec = dict(task)
                rec["prompt_version"] = variant_name
                rec["prompt"] = build_prompt(task, variant_name, cfg)
                rec["ablation_variant"] = variant_name
                rec["evidence_includes_api"] = cfg["use_api"]
                rec["evidence_includes_test"] = cfg["use_test"]
                rec["evidence_includes_code"] = cfg["use_code"]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(json.dumps({"variant": variant_name, "tasks": len(tasks), "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
