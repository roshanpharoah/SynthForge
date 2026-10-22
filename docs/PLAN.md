# SynthForge — Build Plan

**Status date:** 2026-10-23
**Target:** an installable, local-first synthetic data platform. Real ingestion, real models, real
privacy accounting, real evaluation, real jobs, real API/SDK/CLI, real dashboard.

> Scope statement: SynthForge is a research/engineering tool. It is **not** a medical device and makes
> no clinical, legal, or regulatory compliance claim. DP guarantees are stated only for the exact
> mechanism, data range, and composition accounted for in the run manifest.

## Environment discovered

- Empty workspace (`C:\Users\phara\Downloads\synthetic data`).
- Python 3.13.0 only (the brief requests 3.11; the package supports >=3.10 and is verified on 3.13).
- Torch 2.6.0 CPU, NumPy, pandas, SciPy, scikit-learn, FastAPI, Pydantic v2 available.
- Node 22 / npm 10 available for the dashboard build.
- Docker is **not installed**; Docker artifacts will be delivered but the build is NOT RUN.
- Opacus 1.6 installed for an optional DP-SGD cross-check test.

## Design principles

1. **One product path.** Simulators write real files/DBs. Demo uses the same ingestion, job, model,
   evaluation, and registry code as bring-your-own-data (BYO).
2. **No invented numbers.** Every metric is computed; every run has a manifest; UI reads API only.
3. **Honest guarantees.** DP-SGD is implemented per-sample. CTGAN's DP claim covers the critic;
   the generator is post-processing. Preprocessing statistics are either DP or explicitly excluded.
   Empirical attacks are labelled complementary evidence, not a proof.
4. **Fail loudly, usefully.** Typed exceptions with remediation hints; no silent coercion of real data.
5. **CPU-first.** Fast profile trains small real models in seconds/minutes; full profile scales.

## Package map

```
synthforge/
  core/          protocol, config, determinism, manifest, artifact store, registry, errors
  ingest/        tabular, database, timeseries, imaging, profiling
  metadata/      schema inference, constraints, sensitivity
  models/        preprocessing; tabular (CTGAN, TVAE, copula, noise), relational, ECG, imaging, multimodal
  privacy/       DP-SGD, RDP accountant, budget manager, attacks, k-anonymity
  eval/          fidelity, utility, validity, medical metrics, scorecard, reports
  simulator/     seeded banking, healthcare, ECG, MRI, enterprise sources
  jobs/          SQLite-backed queue, worker, handlers, telemetry
  registry/      versioned filesystem artifacts + checksums
  api/           FastAPI, auth, RBAC, rate limit, audit
  sdk/           local Python SDK
  cli/           Typer commands
  ui/            React + TS + Vite + Tailwind + Recharts (built into package)
  db/            SQLAlchemy metadata DB, Alembic migrations
tests/           unit, property, contract, end-to-end
docs/            architecture, decisions, privacy, API, user guide, FDE playbook, benchmarks
```

## Phases and gates

| Phase | Deliverable | Gate |
|---|---|---|
| P1 | skeleton, config, seeding, manifest, storage, DB, simulators | import, simulator tests, ruff/mypy/pytest |
| P2 | CSV/TSV/Parquet/Excel/JSONL/SQL ingest, schema, QI profile, constraints | malformed-input tests, round trip |
| P3 | preprocessing + CTGAN/TVAE/copula/noise | fit/sample on unseen schema; round-trip |
| P4 | DP-SGD + RDP + budget | grad clip/noise tests, accountant reference, budget refusal |
| P5 | relational hierarchy + integrity + exporters | zero orphan FK, unique PK |
| P6 | ECG + imaging generators | shape/conditioning/physiology smoke |
| P7 | evaluation + attacks + model cards + reports | sanity metrics, memorization flag |
| P8 | jobs + API + SDK + CLI | E2E API upload→train→sample→eval→download |
| P9 | dashboard | real API build/serve, no mock adapter |
| P10 | packaging, Docker, CI, docs, benchmarks | clean install, final verification report |

After each phase: `ruff check`, `mypy synthforge`, `pytest`, then git commit.

## Profiles

- `fast`: tiny epochs, small nets, 2 seeds max, single-process; used by tests/CI/demo.
- `full`: larger nets/epochs, more seeds, full-resolution media. Never required for correctness.

## Privacy statement template

Every run model card contains: mechanism, ε, δ, sampling rate, steps, clipping norm, noise multiplier,
**covered** computation, **excluded** computation, assumptions, and whether preprocessing was DP.
Non-DP runs say so explicitly and report no ε.

## Explicit cuts (only if forced, in this order)

3D MRI training → IoT/sequence demo → Postgres polish → dark mode → Compare polish.
Never cut: ingestion, DP-SGD/accountant, attacks, evaluation, relational integrity, jobs/persistence,
API/SDK/CLI, tests, docs.
