# Repository Presentation Audit

This audit records the repository state before the documentation polish pass. It
focuses on presentation, navigation, setup clarity, and
reproducibility. It does not recommend changes to algorithms, experimental
logic, or scientific conclusions.

## Current Structure

```text
.
|-- AGENTS.md
|-- Makefile
|-- README.md
|-- configs/
|   |-- experiments/
|   `-- models/
|-- outputs/
|-- pyproject.toml
|-- reports/
|-- scripts/
|-- src/ttc_operatorbench/
|-- tests/
`-- uv.lock
```

The GitHub repository root also contains a top-level `README.md`, `LICENSE`, and
the nested project directory `ttc-operatorbench/`.

## Strengths

- The package has a modern `pyproject.toml` with Python version bounds,
  dependencies, a console-script entry point, Ruff, mypy, and pytest
  configuration.
- The repo includes a `uv.lock`, which improves install reproducibility.
- The `Makefile` exposes a clear `make check` target for Ruff, mypy, and pytest.
- Experiment protocols are config-driven under `configs/experiments/`.
- Default checks and toy protocols are local and deterministic; real Hugging
  Face model runs are explicitly gated by `RUN_REAL_MODEL_TESTS=1`.
- Outputs are organized under `outputs/`, reports under `reports/`, and those
  generated directories are ignored except for `.gitkeep` files.
- Tests cover schemas, tasks, verifier behavior, search policies, metrics,
  reports, and mocked Hugging Face provider behavior.

## Presentation Gaps Found

### Missing Documentation

- No `docs/` directory existed before this pass.
- The project lacked dedicated pages for project overview, experiment design,
  reproducibility, and result interpretation.
- The GitHub-facing root README only pointed to the nested project README, so a
  reader landing on the repository had to click through before seeing setup,
  purpose, or quickstart information.

### Setup Clarity

- The correct setup route is `uv sync --all-groups`, but the previous README did
  not explain why `uv` is the supported path or what Python versions are
  expected.
- The README did not explicitly say that default checks do not require model
  downloads, network inference, GPU access, or Hugging Face credentials.
- The Makefile defaults to Python 3.12. That is valid, but users relying only on
  `python3` may need to verify that their shell resolves to Python 3.11 or 3.12.

### Experiment Command Clarity

- The default config-driven command was present, but the README did not give a
  short sequence from setup to checks to the smallest local
  experiment.
- The relationship between `scripts/run_toy_eval.py`,
  `scripts/run_experiment.py`, `scripts/make_plots.py`, and
  `scripts/make_portfolio_report.py` was not laid out in one place.
- Real-model commands were correctly gated, but they would benefit from a clearer
  warning that they may download model weights and are not part of default
  verification.

### README Weak Sections

- Missing explicit research question.
- Missing repository tree.
- Missing concise "Quickstart".
- Missing capabilities section connecting the harness to
  test-time compute, verifier-guided reasoning, adaptive operator selection, and
  budgeted evaluation.
- Current limitations were present in spirit, but they were separated from the
  quickstart path and could be made easier to scan.

### Reproducibility Instructions

- The previous README described output locations for the default protocol, but
  it did not list every generated artifact in a compact guide.
- Seeds were visible in config files but not summarized in documentation.
- The public/hidden test separation was mentioned, but deserved a more explicit
  reproducibility and research-integrity explanation.

### Results and Outputs

- Generated files such as `attempts.jsonl`, `search_results.jsonl`,
  `summary.json`, `summary.csv`, `decision.json`, success plots, and the Markdown
  report were named, but not explained by role.
- The meaning of the current `needs_analysis` verdict was not easy to find for a
  reader.
- Portfolio report generation existed but needed a clearer description of when
  to use it.

### Potentially Confusing Files

- The nested project layout means there are two READMEs. This is workable, but
  the root README should act as a polished front door and point clearly into the
  package directory.
- `scripts/.gitkeep` remains in a non-empty scripts directory. This is harmless,
  but visually unnecessary.
- `configs/models/model_roster.yaml` includes future or optional model entries.
  Documentation should clarify that these are not all enabled or validated in
  default runs.

## Safe Documentation Additions Made

- Added this audit.
- Added `docs/project_overview.md`.
- Added `docs/reproducibility.md`.
- Added `docs/experiment_design.md`.
- Added `docs/results_guide.md`.
- Expanded the package README into a public research-repo README.
- Expanded the GitHub root README so the repository is understandable from the
  landing page.

## Remaining TODOs

- Add a formal citation only if there is a paper, preprint, or archived release
  to cite.
- Add exact hardware/time expectations for real-model runs after they are
  measured on the intended local or cloud setup.
- Add a final benchmark report only after larger real-model protocols are run
  and reviewed.
- Decide whether to remove `scripts/.gitkeep` in a future cleanup-only commit.
