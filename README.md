# Lantern-Fly-Predictor

Group Members:

Tejas Reddy - zkw2ag

Lukas Vrouwenvelder - kuc3by

Nasrullah Safdari - hdq6xy

Michael Wagner - vfb8rk

Aaron Witzel - asy3fx

**Problem Space:**

We aim to create and compare prediction models (ones learned in class and outside the scope of the class) on Lantern Fly population in the US given a set of predictor variables describing the environment of these lantern flies to determine which factors affect their growth in population the most.
The predictor variables we will be using are:

- Latitude
- Longitude
- Climate (Temperature/Humidity/Sunlight/Rainfall)
- Host Plants (Categorical)
- Season
- Propogation Timeline (rate at which the Lantern Fly propogates)

Note: This is not the final list and is subject to change depending on available datasets in this space.

Some of the available datasets found so far are from sources such as 

- iNaturalist (Zenodo) - Lantern Fly historical sightings/population data
- USDA Forest Service Maps (FIA DataMart)
- Tree of Heaven distribution maps (eddmaps through University of Georgia)
- PRISM/Daymet (Temperature, Precipitation, other climate values)




---

## Project Roadmap

> _Drafted with AI assistance (Claude Code) from team notes and the DS6021 rubric; reviewed and edited by the team. Logged in the AI-usage documentation._

### Deliverables (due Sun 12/6, presentations 12/7)
- Slides (8–10, PDF/HTML) with a live app demo; 12 min including Q&A
- Raw **and** cleaned data, plus the code that turns one into the other
- Reproducible, commented GitHub repo with run instructions
- Publicly deployed app (Streamlit, approved by the professor)
- AI-usage document: AI-generated code flagged and how it was verified; any CLAUDE.md/prompts included. **No AI for written explanations or reflections.**

### Open Decisions (resolve before modeling)
1. **Response variable.** What does "population" mean given our data? Sightings are presence-only and biased by observer effort. Options: presence/absence, effort-adjusted report counts, spread/first detection, or true abundance (if survey data can be found).
2. **Unit of analysis.** Point, grid cell, or county; time step (year, season, month).
3. **Scope.** Geographic extent (invaded region vs. all US) and years covered.
4. **Primary goal.** Prediction (forecast/map risk) vs. inference (which factors matter), or how both will be reported.
5. **Stakeholder and cost of error.** Who uses the prediction, and is a missed infestation worse than a false alarm? This drives the evaluation metric.
6. **Spread vs. environment.** Whether to include proximity/time-since-arrival predictors, which help forecasting but can mask environmental effects.
7. **Predictors.** Final list, at least 4 numeric and 4 categorical, aligned to the same spatial/temporal unit.
8. **Hypotheses.** State expected relationships (e.g., winter temperature limits establishment) before testing.

### Methodology Principles
- **Validation** reflects the use case: hold out later years and/or regions rather than random splits; compare against a simple baseline.
- **Multiple models** from the course (e.g., GLMs, regularized regression, KNN, tree-based), with the best clearly justified.
- **Unsupervised learning** (PCA and/or clustering) is attempted and either used or rejected with explanation.
- **Assumptions checked** (e.g., overdispersion, multicollinearity, spatial dependence) and discussed.
- **Experiments tracked** in one consistent, reviewable place.
- **Data preparation** documented: cleaning, missing data/imputation, transformations, and feature engineering tied to the problem.

### Timeline
| Week | Dates | Milestone |
|---|---|---|
| 9 | 10/19–10/23 | Resolve open decisions; repo and environment setup |
| 10 | 10/26–10/30 | Data acquisition, cleaning, EDA; first baseline model |
| 11 | 11/2–11/6 | Feature engineering; app scaffold (end-to-end skeleton) |
| 12 | 11/9–11/13 | At least two modeling approaches compared |
| 13 | 11/16–11/20 | PCA/clustering integrated; model refinement |
| 14 | 11/23–11/27 | App deployed; conclusions drafted |
| 15 | 11/30–12/6 | Slides, rehearsal, final repo cleanup, submission |

### Roles
_To be assigned:_ data pipeline · EDA/visualization · modeling · unsupervised learning · app · presentation/AI documentation.

### Repository Structure
```
data/raw/        original downloads (not committed; re-created by src/data_sources.py)
data/processed/  cleaned, model-ready data
notebooks/       EDA and modeling notebooks
src/             reusable data and modeling code
experiments/     experiment log (every model run, same train/test rows)
app/             Streamlit dashboard and its small data files
presentations/   project walkthrough slides (open the .html in a browser)
docs/            AI-usage log, to-do list, notes
```

## Getting Started
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/jupyter lab notebooks/01_data_loading.ipynb
```
Run the notebooks in order: `01_data_loading` (data), `02_eda` (EDA), then `03_regularized_regression` (first model). The first run downloads the data into `data/raw/` (about 10 minutes, mostly the iNaturalist API). Later runs use the cached files. AI usage is logged in [`docs/ai_usage.md`](docs/ai_usage.md).

## Modeling Table
All models use one shared county-year table: 8,790 rows (train 2015–2022: 5,309; test 2023–2025: 3,481), 36 columns.
- **Saved copy:** [`data/processed/model_table.csv`](data/processed/model_table.csv) (also `.parquet`), with the row count after each filter in `model_table_filter_steps.csv`.
- **Source of truth:** `build_model_table()` in [`src/modeling.py`](src/modeling.py), which shows every merge, filter and feature definition. Feature sets A, B and C are listed in `FEATURE_SETS`.
- **Use in a notebook:** `df = m.load_model_table()`, then `train, test = m.split(df)`. Log results with `m.log_experiment(...)`.
- **Rebuild after changing the code or the processed data:** `cd src && python modeling.py`

## Streamlit App
The dashboard in [`app/`](app/) has four pages: a county map explorer (sightings, effort, predictions and climate by year), actual vs. predicted sightings for any county, a model comparison built from [`experiments/experiment_log.csv`](experiments/experiment_log.csv), and an About page.

Run it locally from the repo root:
```bash
.venv/bin/streamlit run app/streamlit_app.py
```

The app reads only the small files in `app/data/` (committed, about 1 MB) plus the experiment log, so it needs no raw data or internet. After the modeling table or the chosen model changes, rebuild those files and commit them:
```bash
.venv/bin/python app/prepare_data.py
```

**Deploying (Streamlit Community Cloud, free):** sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub, create an app from this repo, branch `main`, main file `app/streamlit_app.py`. It installs from [`app/requirements.txt`](app/requirements.txt) and redeploys on every push to `main`.
