# Generative AI Usage Disclosure

Required by the DS6021 final project rubric. AI-generated code is marked with an
`# AI-GENERATED` header (or a note in the notebook's first cell). 

Written explanations/interpretations/reflections are done by hand. 

| Date | Tool | What | Files | How it was verified |
|---|---|---|---|---|
| 2026-10-27 | Claude Code (Claude Opus 5.5) | Data download/loading code and first notebook | `src/data_sources.py`, `notebooks/01_data_loading.ipynb`, `requirements.txt` | See below |
| 2026-10-28 | Claude Code (Claude Opus 5.5) | GBIF observer-effort loaders; EDA section 1 (observer effort) | `src/data_sources.py` (GBIF section), `notebooks/02_eda.ipynb` | See below |
| 2026-11-03 | Claude Code (Claude Opus 5.5) | Preliminary regularized regression (Ridge / Lasso / Elastic Net) | `notebooks/03_regularized_regression.ipynb`, `requirements.txt` (added scikit-learn) | See below |

## Verification: 2026-10-27 data loading
- **iNaturalist:** 98,859 rows vs. 98,856 from the API's `total_results` at the start (3 new sightings during the download); 0 duplicate IDs. The first attempt died silently at 23,800 rows, so the download was made resumable with retries.
- **NOAA climate:** spot-checked Philadelphia (42101) 2023 monthly averages against expected seasonal values (Jan ~42 °F, Jul ~79 °F). Checked min/max of every column after parsing, which caught that missing temperatures are `-99.90` rather than the documented `-99.99`.
- **County join:** compared climate FIPS codes with Census FIPS codes. Found and fixed the Connecticut mismatch (switched to 2020 boundaries) and documented the remaining gaps (DC, Lexington VA, NOAA code 24511).
- **Still to do by the team:** review the code and confirm the iNaturalist query filters (research grade, US) are the ones we want.

## Verification: 2026-10-28 observer effort
- **GBIF taxon key** 5157899 confirmed as *Lycorma delicatula* via `/species/5157899`.
- **GADM → FIPS county matching:** every lower-48 Census county checked for a match. Fixed accented names (Doña Ana), GADM labeling Fairfax City as a county, and two renamed/merged counties. Four cities that GADM merges into their same-named county are documented as a limitation.
- **Counts:** GBIF lanternfly total (~70k) is consistent with GBIF holding ~73% of the iNaturalist API's records.
- **Charts** were checked visually against the printed tables.

## Verification: 2026-11-03 regularized regression
- **Response variable:** the first choice was the log rate per 10k observations. Before modeling, the EDA 1b log-log slope was recomputed from the saved data: 0.51 (r = 0.39, n = 1,080). Dividing by effort assumes a slope of 1, so the response was changed to `log1p(slf_obs)` with `log(all_obs)` as a predictor. The notebook reproduces the 0.51 (0.57 on the modeling table).
- **Modeling table:** row counts printed at each filter step (35,827 → 8,790). The 15 rows dropped for missing climate are DC and Lexington, VA, both already-known NOAA gaps. Asserts confirm no missing values in the predictors or the response.
- **Leakage checks:** an assert confirms train (2015–2022) and test (2023–2025) years don't overlap. The lag feature is built by shifting the year forward by 1, so it only uses the previous year. CV folds are expanding-window by year (printed in the notebook).
- **Penalty grid:** on the first run, feature set A chose the smallest alpha on the grid. The grid was widened from 1e-4 to 1e-6 and A still chose the smallest value, so the notebook flags this with an `alpha_at_grid_edge` column and a note that A is effectively OLS. Feature set B's alphas are inside the grid.
- **Baselines:** every model is compared with the training mean, effort-only OLS and persistence. On the test years, persistence has a lower MAE (0.214) than the best model, ElasticNet (B) (0.251), even though the best model's RMSE is lower (0.525 vs. 0.556). The table reports both.
- **Charts** were checked visually against the printed tables (coefficient bars against the coefficient table, the CV alpha line against the chosen alpha).
- **Run:** executed end to end in the `ds6021` conda environment (Python 3.12, scikit-learn 1.9.0) with no errors.
- **Interpretation** cells are left as `_TODO_` for the team.

## Prompts
Prompts were given conversationally in Claude Code. Key requests:
1. "Given that description, what other questions need to be answered to define the project scope and objectives?"
2. "Condense questions and direction in a README form so that we have a roadmap for the project."
3. "Work on getting data loaded into the notebook" (iNaturalist + climate; Python with requirements.txt).
4. "Looking at the data, what do you see as the best ways to continue EDA?" then "Let's work on these 1 at a time" (started with observer effort).
5. "Make a new notebook that runs a preliminary regression model with regularization on the dataset in the EDA notebook" (scope, feature sets and response chosen by the team in follow-up questions; the response was changed after asking "why are we using log rate here?").
