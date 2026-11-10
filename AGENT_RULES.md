# Agent Rules

- Read this file and only the context file needed for the current task before editing.
- Do not reload every source file or dataset into the conversation. Inspect the specific module you will change.
- Reuse `src/` and `configs/experiment_config.yaml`. Do not create a second training path.
- Keep new functions small and in the existing module that owns that step.
- Add a dependency only if the current pipeline cannot do the job.
- Keep seeds, sample sizes, and paths in the YAML config. Paths stay project-relative.
- Do not invent metrics, citations, or dataset counts. Read `results/metrics/summary.json` and `data/metadata/*_processed.json`.
- Update `reports/` and the context file that changed when a method or result changes.
- Do not overwrite `results/` or `models/` without recording the reason in `reports/experiment_log.md`.
- Do not claim temporal, cross-dataset, drift, or adaptation results unless that experiment was run and saved.
- Do not put labels, dates, identifiers, HTML, or target-derived similarity scores (e.g. `URLSimilarityIndex`) into model features.
- Never evaluate temporal shift on months lacking both classes (e.g. Nov 2024–Apr 2025 in PhreshPhish).
- Fit preprocessing on the training split only.
- Leave the PhiUSIIL zip unchanged. Do not download PhreshPhish HTML for routine runs.
- The final methodology audit (`reports/final_methodology_audit.md`) and authoritative index (`results/final/final_results_summary.csv`) are the primary authorities for metrics.
- The final scientific verdict is: **Scientifically defensible with documented limitations.** Do not claim unconditional full validation.
