# SynthForge

Local-first enterprise synthetic data platform: ingest real-world schemas, train generative
models, apply differential privacy, evaluate fidelity/utility/privacy, and export governed
synthetic datasets.

> **Status: research/engineering prototype.** The library, ingestion, models, evaluation,
> privacy accounting, budget ledger, and reporting are implemented and smoke-tested against
> bundled **simulated** data. The HTTP API, CLI entrypoints, web dashboard, and container
> images are **not implemented in this snapshot**. See [Current status](#current-status).

---

## What works today

| Area | Capability |
| --- | --- |
| Ingestion | CSV/TSV/Parquet/Excel/JSONL, SQLAlchemy databases (SQLite default, Postgres optional), relational discovery with orphan-FK and cycle reporting, WFDB/EDF/CSV/NumPy ECG, NIfTI/DICOM imaging with PHI stripping |
| Simulation | Seeded, documented simulators for banking, healthcare, 12-lead ECG, MRI phantoms, event/IoT data (all labelled **SYNTHETIC SOURCE**) |
| Models | VGM-style preprocessing, CTGAN (conditional, PacGAN, WGAN-GP), TVAE, Gaussian copula, resample+noise baselines, hierarchical relational synthesis, ECG conditional generator, Diffusion-GAN imaging |
| Privacy | From-scratch DP-SGD (per-sample clipping + Gaussian noise + Poisson sampling), RDP accountant, ε-budget manager with ledger, empirical attack suite, k-anonymity |
| Evaluation | KS, chi-square/TVD, Wasserstein, correlation/MI differences, discriminator AUC, TSTR vs TRTR, FID-lite, precision/recall, physiological validity, scorecard, HTML/JSON/PDF reports |
| Governance | Model cards, datasheets, reusable manifests, checksummed artifacts, audit/budget tables |

**Privacy scope is explicit.** Only CTGAN has an accounted DP path (critic training).
Generator sampling is post-processing. Profiling, evaluation, and attacks are excluded.
TVAE, copula, resample+noise, relational, ECG, and imaging runs are recorded as
**non-private** and state that in their privacy report. No claim of clinical, diagnostic,
legal, or regulatory validity/compliance is made.

## Requirements

- Python 3.10+ (developed on 3.13; the packaging target is 3.10+)
- CPU is sufficient; CUDA is optional
- No API keys, external services, or pretrained weight downloads

## Install

```cmd
cd "C:\Users\phara\Downloads\synthetic data"
pip install -e ".[dev]"
```

Or run from source without installing:

```cmd
set PYTHONPATH=%CD%
```

## Quick start (library)

```python
from pathlib import Path

from synthforge.core.config import Profile, RunConfig
from synthforge.ingest.tabular import load_csv
from synthforge.models.registry import build_model
from synthforge.simulator.banking import write_unseen_schemas

root = Path("_demo")
write_unseen_schemas(root)

dataset = load_csv(root / "survey_unseen.csv", profile=True)
config = RunConfig(model="ctgan", profile=Profile.FAST, seed=42).apply_profile()

model = build_model("ctgan", config)
model.fit(dataset, config)
synthetic = model.sample(500, seed=7)

print(synthetic.primary_table.frame.head())
```

## Differential privacy (CTGAN)

```python
config = RunConfig.model_validate({
    "model": "ctgan",
    "profile": "fast",
    "privacy": {"enabled": True, "epsilon": 2.0, "delta": 1e-5},
}).apply_profile()
```

The run reports the exact ε consumed, δ, noise multiplier σ, step count, and what the
guarantee covers.

## Evaluation

`synthforge.eval.suite.evaluate_dataset(real, synthetic, attacks=True)` returns per-table
fidelity, utility, validity, attack results, re-identification screening, and a scorecard.
`synthforge.eval.report.write_report(...)` writes HTML, JSON, printable HTML, and PDF.

## Privacy budget

`synthforge.privacy.budget.BudgetManager` stores per-project and per-dataset ε budgets in
SQLite and refuses runs that would exceed them.

## Current status

Implemented and verified by smoke tests on bundled simulated data:

- ingestion (tabular, relational, ECG, imaging)
- tabular, relational, ECG, and imaging model fit/sample
- CTGAN DP-SGD end to end (ε≈1.99 for a 2.0 target at δ=1e-5)
- evaluation, reports, attack suite
- budget manager and RDP accountant
- RDP monotonicity and noise calibration

Not yet implemented in this snapshot:

- `synthforge/cli/main.py` (the `synthforge` console entrypoint is declared but missing)
- `synthforge/api/main.py` (FastAPI app)
- web dashboard
- Docker/Compose images and CI workflows
- formal pytest suite
- Alembic migrations
- benchmark documents

`ruff check synthforge` passes. `mypy synthforge` reports remaining type errors.

## Repository layout

```
synthforge/
  core/        configuration, types, manifests, workspace, artifact store
  db/          SQLAlchemy models and session handling
  ingest/      tabular, database, time-series, imaging ingestion + profiling
  metadata/    schema inference and constraint discovery/repair
  simulator/   seeded simulators for all supported modalities
  models/      preprocessing and generative models
  privacy/     DP-SGD, RDP accountant, budget manager, attack suite
  eval/        fidelity, utility, validity, medical metrics, scorecard, reports
  registry/    model registry
docs/          design notes and build plan
```

Layer boundaries are intentional: `core/` does not import API or UI code.

## Data and privacy notices

- Bundled data is generated by seeded simulators and labelled **SYNTHETIC SOURCE**.
  It does not describe real people, accounts, or examinations.
- Do not ingest identifiable data without the appropriate legal basis and controls.
- Differential privacy guarantees are mathematical statements about the training mechanism,
  not statements about downstream use, re-identification by other means, or regulatory
  compliance.

## License

Apache-2.0. See `pyproject.toml`.
