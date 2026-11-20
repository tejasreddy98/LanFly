# Generative AI Usage Disclosure

Required by the DS6021 final project rubric. AI-generated code is marked with an
`# AI-GENERATED` header (or a note in the notebook's first cell). 

Written explanations/interpretations/reflections are done by hand. 

| Date | Tool | What | Files | How it was verified |
|---|---|---|---|---|
| 2026-10-27 | Claude Code (Claude Opus 5.5) | Data download/loading code and first notebook | `src/data_sources.py`, `notebooks/01_data_loading.ipynb`, `requirements.txt` | See below |
| 2026-10-28 | Claude Code (Claude Opus 5.5) | GBIF observer-effort loaders; EDA section 1 (observer effort) | `src/data_sources.py` (GBIF section), `notebooks/02_eda.ipynb` | See below |
| 2026-11-03 | Claude Code (Claude Opus 5.5) | Preliminary regularized regression (Ridge / Lasso / Elastic Net) | `notebooks/03_regularized_regression.ipynb`, `requirements.txt` (added scikit-learn) | See below |
| 2026-11-03 | Claude Code (Claude Opus 5.5) | lydemapr survey loader; iNaturalist vs. field-survey comparison (EDA 02b) | `src/data_sources.py` (lydemapr section), `notebooks/02b_inat_vs_lydemapr.ipynb`, `requirements.txt` (added scipy, statsmodels) | See below |
| 2026-11-03 | Claude Code (Claude Opus 5.5) | FIA host-tree and GBIF Tree of Heaven loaders; host-plant EDA (02c) | `src/data_sources.py` (host plants section), `notebooks/02c_host_plants.ipynb` | See below |
| 2026-11-17 | Claude Code (Claude Opus 5.5) | County categorical features: rural-urban codes, EPA ecoregions, interstates, rail, Census division (EDA 02d) | `src/data_sources.py` (county features section), `notebooks/02d_county_features.ipynb`, `requirements.txt` (added xlrd) | See below |
| 2026-11-17 | Claude Code (Claude Opus 5.5) | Shared modeling table, experiment log, negative binomial model (notebook 04) | `src/modeling.py`, `notebooks/04_negative_binomial.ipynb`, `experiments/experiment_log.csv` | See below |
| 2026-11-20 | Claude Code (Claude Opus 5.5) | Streamlit app skeleton (map explorer, county predictions, model comparison) and its data prep | `app/streamlit_app.py`, `app/prepare_data.py`, `app/data/`, `app/requirements.txt`, `requirements.txt` (added streamlit, plotly) | See below |

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

## Verification: 2026-11-03 iNaturalist vs. lydemapr (02b)
- **Source checked:** Zenodo record 7976229 (v1.0.0, 2023). Column definitions taken from its `_Metadata` file. Inspected every source and collection method before filtering.
- **Independence:** lydemapr includes 11,181 iNaturalist rows and 269,380 public-hotline reports. Both are dropped; only field surveys (PDA, other state agencies, USDA; 377,829 records, 2014–2021) are used as the reference.
- **Parsing:** `lyde_established` loads with mixed types, so it is read as a nullable boolean (78k missing values kept as missing, not counted as False). Verified every `lyde_present == False` record has density "Unpopulated".
- **County join:** 0.18% of survey points fall outside any county.
- **Errors fixed during the run:** nullable booleans broke the mean aggregation, and SciPy's `spearmanr` failed on pandas columns under NumPy/Python 3.14; both converted to plain floats/arrays. Results were rechecked after each fix.
- **Thresholds are choices, not findings:** ≥10 survey points per county-year and ≥5% established for "established". The notebook states both; results may change with different cutoffs.
- **Charts** checked against the printed tables. The "High" density group has n = 3 and is flagged as too small to interpret.

## Verification: 2026-11-03 host plants (02c)
- **FIA API:** the parameter pages don't list the evaluation codes in a usable form, so the most recent evaluation group per state was parsed from the `wc` table and a single-state query (Delaware) was inspected by hand before running all 48 states. Berks County, PA was spot-checked (maples ~19% of live trees, Tree of Heaven present).
- **FIA species codes:** Tree of Heaven 341, black walnut 602, maples 313–322, willows 920–929 (FIA species list). Standard errors within a host group are combined assuming independence, which is noted in the code as an approximation.
- **Coverage checked:** 2,724 of 3,108 lower-48 counties have FIA data. The gaps (west Texas/Plains, 34 Virginia independent cities including Arlington) are reported in the notebook.
- **Noise checked:** median relative SE of Tree of Heaven estimates is 93%. The notebook states that FIA zeros mostly mean "not sampled on forest plots".
- **GBIF taxon key** 3190653 confirmed as *Ailanthus altissima* (exact, accepted) via `/species/match`.
- **Caveats added after reviewing results:** geographic confounding with distance from PA, and shared reporting between lanternfly and Tree of Heaven campaigns inflating the GBIF correlation.
- **Run:** both notebooks executed end to end in the project `.venv` (Python 3.14) with no errors.
- **Interpretation** cells are left for the team.

## Verification: 2026-11-17 county features (02d)
- **Sources checked before use:** every download URL was tested; the USDA plant hardiness zone file was unavailable (404) and was dropped, since it would largely duplicate winter temperature.
- **Rural-urban codes:** 8 counties were missing after the join, all in Connecticut, because the 2023 file uses CT's new planning regions. Filled from the 2013 edition (population 2010) and flagged in `rucc_source`. Final table has no missing values (3,108 lower-48 counties).
- **Spot checks:** Manhattan, Arlington and DC are code 1 (large metro) with interstates and rail; Berks County, PA is code 2 in the Ridge and Valley ecoregion.
- **Geometry:** lengths and areas computed in CONUS Albers (EPSG:5070). Ecoregion boundaries simplified at 500 m before taking each county's largest-area ecoregion.
- **Finding flagged in the notebook:** Level I ecoregion has only 2 testable groups in the invasion region, so Level III is the usable version.
- **Run:** executed end to end in the project `.venv` with no errors. Interpretation cells left for the team.

## Verification: 2026-11-17 modeling table and negative binomial (04)
- **Shared table reproduces notebook 03 exactly:** same filter steps and row counts (35,827 → 8,790; train 5,309, test 3,481), persistence baseline RMSE 0.556 / MAE 0.214, and Elastic Net (B) re-fitted with notebook 03's settings scores RMSE 0.525 / MAE 0.251, matching the reported values.
- **Leakage:** the Tree of Heaven sighting rate added in feature set C uses training years (≤2022) only; scalers are fitted on training years only.
- **Fitting problems found and fixed:** the first optimizer (BFGS from default start) failed to converge for feature set B and produced infinite α. The model is now started from the Poisson solution with Newton's method, falling back to BFGS; all three feature sets converge (reported in the notebook).
- **Scoring choice checked:** log1p of the NB predicted mean made the model look much worse than its training fit implied, because high α drags the mean up. The notebook scores the simulated expected log1p(count) instead (4,000 draws, fixed seed), which is the fair counterpart to models fitted on log1p(count). Both are explained in the notebook.
- **Claims checked against data:** largest test-year count (1,723) verified before being quoted; zero shares and per-year errors computed directly.
- **Result reported as is:** NB trails Elastic Net and persistence on every test year; the notebook diagnoses why (distribution shift, explosive extrapolation, collinear climate coefficients) rather than tuning until it looks better.
- **Run:** executed end to end in the project `.venv` with no errors. Interpretation cells left for the team.

## Verification: 2026-11-20 Streamlit app (app/)
- **Model reproduced:** `app/prepare_data.py` refits Elastic Net (B) with notebook 03's alphas, l1 ratios and expanding-window CV on the shared table; test scores (RMSE 0.5248, MAE 0.2513, R² 0.8345) match the experiment log exactly.
- **Bug found and fixed:** the first export merged county names twice (duplicate `state_x`/`state_y` columns), which crashed the app on load; fixed by using the names already in the county-year table.
- **Pages tested:** Streamlit's headless `AppTest` loaded all four pages and every map option with no exceptions; the map was also checked visually in a browser.
- **Self-contained deploy:** the app reads only `app/data/` (~1 MB, committed) and `experiments/experiment_log.csv`, never `data/raw` or the network.
- **Written text is a placeholder:** the About page and chart captions were drafted by AI and must be rewritten by the team (course rule on written explanations).

## Prompts
Prompts were given conversationally in Claude Code. Key requests:
1. "Given that description, what other questions need to be answered to define the project scope and objectives?"
2. "Condense questions and direction in a README form so that we have a roadmap for the project."
3. "Work on getting data loaded into the notebook" (iNaturalist + climate; Python with requirements.txt).
4. "Looking at the data, what do you see as the best ways to continue EDA?" then "Let's work on these 1 at a time" (started with observer effort).
5. "Make a new notebook that runs a preliminary regression model with regularization on the dataset in the EDA notebook" (scope, feature sets and response chosen by the team in follow-up questions; the response was changed after asking "why are we using log rate here?").
6. Planning summaries from two other Claude chats (first-model planning; lanternfly data sources and modeling notes) were pasted in and condensed. Decisions made by the team from them: keep everything in Python, compare iNaturalist with lydemapr, and predict population numbers rather than classification.
7. "Compare inaturalist data with lydemapr for now."
8. "Can we get host plant data as well?"
9. "Let's get a few more categorical features in the set."
10. "Before we build spread features, does it make sense to add another model?" then "Set up both [modeling table and experiment log] and start the negative binomial."
11. "Let's work on the dashboard/application portion of the project" (framework chosen by the team: Streamlit, approved by the professor).
