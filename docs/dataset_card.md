# Dataset Card

## Purpose

The benchmark evaluates whether local language models can recover a user-facing use case from source-code evidence, route or API hints, and optional test-path hints.

## Task Format

Each JSONL record contains a task identifier, repository identifier, gold actor, gold use-case name, gold description, gold main flow, API and test hints, the prompt sent to the model, and the code paths used as evidence.

## Evaluation

The package reports exact and substring name match, token-level name and description F1, soft lexical matching, actor matching, path-fidelity metrics, error categories, and slice-level analyses by framework, domain, evidence size, and confidence bin.

## Limitations

The task set focuses on educational, library, student-management, and learning-management systems. Results should be interpreted as evidence about this benchmark scope, not as a universal estimate for all software domains.
