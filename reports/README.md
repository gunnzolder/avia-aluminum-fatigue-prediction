# Report files

🇺🇦 [Українська версія](README.uk.md)

Everything in this directory (except the two READMEs) is **generated** by
`uv run python pipeline.py all` and can be deleted and rebuilt at any time.
This page explains how the numbers are computed and what each file contains.

## How the calculation works

1. **Convert.** The digitized crack-growth handbook is parsed into a tidy table of
   (ΔK, da/dN) points. Alloy/temper/orientation metadata is recovered from the chart
   titles embedded in the workbook; the handbook's own NASGRO/Walker fit coefficients
   are harvested for cross-checking.
2. **Validate.** Points must satisfy ΔK > 0, da/dN > 0, and the physical bound
   **Kmax = ΔK/(1−R) ≤ 200 ksi·√in**, da/dN ≤ 0.1 in/cycle. Violations are excluded
   and reported — they are digitization errors in the source, not real measurements.
3. **Fit.** Each experimental curve (one alloy/orientation/R combination) is fitted to
   the Paris law **da/dN = C·(ΔK)ᵐ** by ordinary least squares in log-log space.
   Near-threshold and fast-fracture tails, which bend away from the power law, are
   trimmed iteratively while the fit's r² improves.
4. **Train.** Three models — Ridge regression, Random Forest, SVR (RBF) — learn the
   mapping *alloy static properties (Ftu, Fty, Fcy, Fsu, elongation, density, E, G, ν)
   + R ratio + series/orientation → (log₁₀C, m)*. They are evaluated under three
   cross-validation schemes: random K-fold, group K-fold by physical experiment, and
   **leave-one-alloy-out (LOAO)** — the honest test of predicting an alloy absent from
   training.
5. **Predict.** For each target alloy (see `targets.yaml`) the models predict (C, m) at
   the requested R ratios. Uncertainty is a q05–q95 spread band: per-tree quantiles for
   Random Forest, bootstrap refits for Ridge/SVR. It is a spread estimate, **not** a
   calibrated confidence interval.
6. **Physics audit.** Inputs and predictions are checked for positivity, monotonicity
   of da/dN vs ΔK, and plausibility of the Paris exponent for aluminum alloys
   (PASS / WARN / FAIL). FAILed curves are excluded from training; FAILed predictions
   would be reported, not hidden.

> Predicted curves are research results, not certification values.

## File guide

| File | Contents |
|---|---|
| `data_quality.txt` | Conversion log: row counts per table, missing values per column, duplicates dropped, and the list of physically impossible points that were excluded (with source pages). |
| `coverage_matrix.csv` | One row per alloy: `n_static_props` (distinct strength properties), `n_material_variants` (AI-Ready material_ids), `n_cg_points` / `n_cg_curves` / `n_cg_datasets` (crack-growth data), `cg_R_min/max` and `cg_dK_min/max` (experimental ranges), `n_sn_points` / `n_sn_kt` (S-N fatigue data), `n_dt_points` (damage-tolerance data). Alloys with static properties but zero curves are the prediction targets. |
| `fit_crosscheck.csv` | Our per-curve Paris exponent `m` vs the handbook's published exponent `ref_n` on the same page, with `m_vs_ref_n_pct` deviation. Loose comparison only: the book's values belong to NASGRO/Walker equation forms fitted over the full curve, ours to the Paris region. |
| `metrics.csv` | Cross-validation metrics: one row per (scheme × model × target) with `r2`, `mae`, `rmse`. Targets are `log10_C` and `m`. Compare schemes to see how much harder the unseen-alloy problem is than interpolation. |
| `predictions.xlsx` | Sheet **predictions** — one row per (target alloy × R × model): `log10_C_pred`, `m_pred`, `C_pred` (= 10^log10_C), `dK_applicable_min/max` (ΔK range where the prediction is considered valid — the median fitted range of same-series training curves), and `q05/q95` interval bounds per target. Sheet **training_fits** — every fitted training curve with C, m, r², ΔK range, points used, and its training-eligibility flag. |
| `ai_ready_dataset.xlsx` | The converted dataset in the AI-Ready multi-sheet structure: `materials`, `mechanical_properties` (long format), `physical_properties`, `fatigue_data` (S-N points), `DT_Data`, plus the new `crack_growth_points` and `figures_metadata` sheets produced by this project. |
| `figures/paris_<alloy>_R<r>.png` | Predicted Paris curve for one target alloy and R ratio: three model curves with q05–q95 spread bands, log-log axes, plotted over the applicable ΔK range. |
| `figures/loao_log10_C.png`, `figures/loao_m.png` | Leave-one-alloy-out predicted-vs-actual scatter per model. Points near the dashed diagonal = good out-of-alloy prediction. |
| `figures/coverage_matrix.png` | Heatmap version of `coverage_matrix.csv`. |

## Reproducing

```bash
uv run python pipeline.py all      # rebuild everything in this directory
uv run python pipeline.py export   # re-export tables/figures only
```
