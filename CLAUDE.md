# CLAUDE.md

## Project: ML Prediction of Fatigue Characteristics of Aviation Aluminum Alloys

Research prototype of an ML system that predicts missing fatigue characteristics of aluminum alloys. The end goal is practical predictions: **Paris law curves (da/dN–ΔK)** and/or **Wöhler (S-N) curves** for selected alloys whose fatigue data is absent from the source files.

Source of requirements: `inbox/Техническое_задание_ML_Aluminum_Fatigue_v2.docx` (technical specification, v2).

## Source Data (`inbox/`)

| File | Role |
|------|------|
| `Aluminum_Data_AI_Ready_mod.xlsx` | Reference **AI-Ready** data structure (target schema). Sheets: `materials`, `mechanical_properties`, `physical_properties`, `fatigue_data`, `DT_Data`. |
| `Fatigue Crack Growth Database.xlsx` | Raw source database to be converted into the AI-Ready format. ~50 paired sheets per figure (`FIGURE B-N; p.XX` + a `p.XX` data sheet), digitized from a crack-growth handbook. |

## Core Tasks

1. **Converter** — transform `Fatigue Crack Growth Database.xlsx` into the `Aluminum_Data_AI_Ready_mod.xlsx` structure.
2. **Data validation** — check units, missing values, duplicates, incorrect and physically impossible values.
3. **Coverage matrix** — per alloy: which materials have static properties, S-N curves, da/dN–ΔK data, and fracture parameters, and which characteristics are missing.
4. **Target alloy list** — agree on a consistent set of target alloys whose missing fatigue characteristics need to be reconstructed.
5. **Baseline models** — implement and compare at least three approaches: **regression**, **Random Forest**, and **SVR** (support vector regression).
6. **Validation schemes** — implement all three:
   - Random Split
   - Group Split by `curve_id` / `dataset_id`
   - **Leave-One-Alloy-Out (LOAO)** — the key test of whether models can predict properties of an alloy never seen in training.
7. **Physics Audit** — sanity-check predictions:
   - da/dN must be positive;
   - da/dN must grow monotonically with increasing ΔK in the Paris region;
   - S-N curve slope must be correct (finite life decreases as stress amplitude increases);
   - no obviously non-physical values.

## Expected Outputs (per target alloy)

- Predicted Paris equation coefficients **C** and **m**, with the da/dN–ΔK curve plotted over its applicable range;
- and/or predicted S-N parameters with the Wöhler curve plotted over an agreed life range;
- separate predictions from regression, Random Forest, and SVR, plus their comparison;
- uncertainty estimate or prediction spread;
- source material features used and the domain in which the prediction is considered valid;
- LOAO results on known alloys temporarily withheld from training.

## Deliverable Files

- Converted AI-Ready Dataset;
- Reproducible Python project / scripts (re-runnable end to end);
- Excel/CSV with predicted coefficients and metrics;
- Plots of Paris and/or Wöhler curves for target alloys;
- Short report: models, validation, limitations, physics audit results.

Report/conference materials: table of target alloys and predicted parameters, prediction curve plots, model comparison, metrics table, and brief conclusions on the applicability of each method.

## Work Stages

| Stage | Expected result |
|-------|-----------------|
| Data preparation | Converter, cleaned AI-Ready dataset, coverage matrix, target alloy list. |
| First models | First predictions from regression / Random Forest / SVR; preliminary Paris and/or Wöhler curves; first metrics and plots. |
| Phase 1 wrap-up | Verified prediction curves for chosen materials, LOAO validation, Physics Audit, tables and figures for the department report and the autumn conference. |

## Project Layout & Commands

Python 3.12 project managed with **uv**; stack: pandas + Parquet + DuckDB + pandera + scikit-learn + Plotly/Streamlit.

```
src/afml/            # package (installed editable via uv sync)
  convert/           #   xlsx → tidy Parquet (chart_meta, crack_growth, ref_fits, ai_ready)
  models/            #   paris_fit, baselines (ridge/RF/SVR), splits, uncertainty
  schemas.py         #   pandera validation + physical plausibility filter
  coverage.py        #   DuckDB coverage matrix
  features.py        #   alloy-property features for ML
  physics.py         #   physics audit (PASS/WARN/FAIL)
  viz.py, export.py  #   plotly figures + report deliverables
pipeline.py          # CLI: convert | fit | train | predict | export | all
app/dashboard.py     # Streamlit dashboard (Coverage / Curves / Models / Physics audit)
targets.yaml         # target alloys + R ratios to predict
data/processed/      # generated Parquet (gitignored, regenerable)
reports/             # generated xlsx/csv/figures (gitignored)
```

Commands:
- `uv sync` — install environment
- `uv run python pipeline.py all` — full pipeline from raw xlsx to report artifacts
- `uv run pytest` — test suite (includes end-to-end checks against the real workbooks)
- `uv run streamlit run app/dashboard.py` — dashboard (requires pipeline outputs)

Data gotchas already handled in code (see module docstrings): alloy identity lives in chart XML titles, not data sheets; R-label sign conventions are inconsistent across sheets; pages 82–90 contain physically impossible digitized points (flagged, excluded); the third column of each data triple is a scaled duplicate.

## Important Notes & Conventions

- **Research-grade, not certification-grade**: predicted curves are a research result; no additional experimental/engineering verification is required. The main question of phase 1 is how reliably models can reconstruct missing characteristics from other alloys' data plus the target material's known properties.
- Language context: the specification and likely reporting are in **Russian**; code, identifiers, and comments should be in English.
- Keep raw source files in `inbox/` untouched; write generated datasets, models, and figures to separate directories (e.g. `data/`, `src/`, `reports/`, `figures/`).
- Preferred stack: Python (pandas, scikit-learn, matplotlib/openpyxl) — the spec explicitly expects a re-runnable Python project or script set.
- When parsing `Fatigue Crack Growth Database.xlsx`, note the paired-sheet structure (figure sheet + data sheet per handbook page) and one stray sheet named `Лист7`.
