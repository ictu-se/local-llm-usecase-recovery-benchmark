# Local LLM Use-Case Recovery Benchmark

This repository contains the replication package for experiments on evidence-grounded recovery of user-goal use cases from source-code evidence. It includes benchmark tasks, saved local-LLM predictions, evaluation scripts, aggregate result tables, and the result figures used by the associated manuscript.

The repository deliberately excludes manuscript sources, journal templates, cover letters, build artifacts, and submission-management files.

## Contents

- `data/tasks/`: JSONL benchmark tasks with prompts, gold use-case fields, API/test hints, and code paths used as evidence.
- `src/usecase_recovery/`: scripts for task construction, Ollama inference, metric computation, path-fidelity evaluation, error taxonomy, and slice analyses.
- `results/runs/`: saved predictions and per-run metrics for 19 experimental runs.
- `results/aggregate/`: manuscript-level aggregate tables regenerated from the saved predictions.
- `figures/`: exported result figures derived from the aggregate results.

## Environment

Python 3.10 or newer is sufficient for all evaluation scripts. No external Python packages are required.

For fresh local inference, install Ollama separately and pull the models you want to evaluate. The saved predictions in `results/runs/` allow the reported metrics to be reproduced without rerunning model inference.

## Reproduce Reported Results

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 src/usecase_recovery/reproduce_reported_results.py
```

The command recomputes per-run metrics from `predictions.jsonl`, rebuilds extended metrics, path-fidelity metrics, error taxonomy tables, slice analyses, and refreshes `results/aggregate/`.

## Run a New Ollama Model

Start Ollama locally, pull a model, then run:

```bash
python3 src/usecase_recovery/run_ollama.py \
  --model qwen2.5-coder:14b \
  --tasks data/tasks/use_case_tasks_user_goal_v4.jsonl \
  --run-name user_goal_v4_new \
  --temperature 0.1 \
  --num-ctx 8192 \
  --resume

python3 src/usecase_recovery/evaluate_use_case_predictions.py \
  --model qwen2.5-coder:14b \
  --run-name user_goal_v4_new

python3 src/usecase_recovery/evaluate_use_case_predictions_soft.py \
  --model qwen2.5-coder:14b \
  --run-name user_goal_v4_new
```

## Notes on Benchmark Construction

The released tasks contain the prompts and gold labels needed to reproduce reported inference and evaluation results. The original candidate source repositories are not vendored here, keeping the replication package compact and avoiding redistribution of unrelated project files. If rebuilding tasks from raw repositories, place annotated labels under `data/annotations/` and checked-out candidate repositories under `data/candidate_repos/`, then run `src/usecase_recovery/build_use_case_tasks.py`.
