# AI-GENERATED (Claude Code, 2026-11-17). Verification steps are recorded in docs/ai_usage.md.
"""Shared modeling table, feature sets, metrics and experiment log.

Every model notebook should build its data with build_model_table() and record
results with log_experiment(), so all models train and test on identical rows
and their results land in one place (experiments/experiment_log.csv).

The table reproduces notebook 03's definitions and filters exactly, then adds
host-plant and county features as extra columns (no change to the rows).
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import data_sources as ds

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "experiments" / "experiment_log.csv"

YEARS = list(range(2014, 2026))        # 2026 excluded: partial year
TRAIN_YEARS = list(range(2015, 2023))  # 2014 has no prior year for the lag feature
TEST_YEARS = [2023, 2024, 2025]
REGION = dict(lon=(-92, -66), lat=(33, 46))
MIN_OBS = 200
CLIMATE = ["tavg_annual", "tmin_winter", "tmax_summer", "precip_annual"]

FEATURE_SETS = {
    # A and B are notebook 03's feature sets.
    "A": ["log_effort", "tavg_annual", "tmin_winter", "tmax_summer", "precip_annual", "lat", "lon"],
}
FEATURE_SETS["B"] = FEATURE_SETS["A"] + ["lag_log_count"]
# C adds host plants and county features (numeric + categorical as 0/1 columns).
FEATURE_SETS["C"] = FEATURE_SETS["B"] + [
    "log_toh_rate_train", "fia_toh_present", "fia_missing",
    "rail_km_per_km2", "interstate_km_per_km2", "is_metro", "has_interstate",
] + ["div_" + d.lower().replace(" ", "_") for d in [
    "Middle Atlantic", "New England", "East North Central", "East South Central",
    "West North Central", "West South Central"]]  # baseline (left out) = South Atlantic


def build_model_table(verbose=True):
    """Return the county-year modeling table (one row per county-year) and the filter steps."""
    cy = pd.read_parquet(ds.PROCESSED / "county_year_effort.parquet")
    cy = cy[cy["year"].isin(YEARS)]
    clim = pd.read_parquet(ds.PROCESSED / "climate_county_month.parquet")

    # Annual climate summaries (same definitions as notebook 03); incomplete years -> NaN.
    g = clim.groupby(["fips", "year"])
    annual = pd.DataFrame({
        "tavg_annual": g["tavg_f"].mean().where(g["tavg_f"].count() == 12),
        "precip_annual": g["precip_in"].sum(min_count=12),
    })
    summer = clim[clim["month"].isin([6, 7, 8])].groupby(["fips", "year"])["tmax_f"]
    annual["tmax_summer"] = summer.mean().where(summer.count() == 3)
    # Winter that ends in year Y = Dec of Y-1 plus Jan-Feb of Y.
    w = clim[clim["month"].isin([12, 1, 2])].assign(year=lambda d: d["year"] + (d["month"] == 12))
    wg = w.groupby(["fips", "year"])["tmin_f"]
    annual["tmin_winter"] = wg.min().where(wg.count() == 3)
    annual = annual.reset_index()

    counties = ds.load_counties()
    cent = counties.to_crs(5070).geometry.centroid.to_crs(4326)
    centroids = pd.DataFrame({"fips": counties["fips"], "lon": cent.x, "lat": cent.y})

    # Last year's lanternfly count, from the full (unfiltered) table.
    lag = cy[["fips", "year", "slf_obs"]].assign(year=lambda d: d["year"] + 1)
    lag["lag_log_count"] = np.log1p(lag.pop("slf_obs"))

    df = (cy.merge(centroids, on="fips", how="left")
            .merge(annual, on=["fips", "year"], how="left")
            .merge(lag, on=["fips", "year"], how="left")
            .fillna({"lag_log_count": 0}))
    df["log_effort"] = np.log(df["all_obs"])
    df["y"] = np.log1p(df["slf_obs"])

    steps = [("all county-years 2014-2025", len(df))]
    df = df[df["lon"].between(*REGION["lon"]) & df["lat"].between(*REGION["lat"])]
    steps.append(("in invasion region box", len(df)))
    df = df[df["all_obs"] >= MIN_OBS]
    steps.append((f"with >= {MIN_OBS} observations", len(df)))
    df = df.dropna(subset=CLIMATE)
    steps.append(("with complete climate", len(df)))
    df = df[df["year"].isin(TRAIN_YEARS + TEST_YEARS)]
    steps.append(("in train + test years (2015-2025)", len(df)))

    df = df.merge(_county_columns(), on="fips", how="left")
    df["split"] = np.where(df["year"].isin(TRAIN_YEARS), "train", "test")
    df = df.sort_values(["year", "fips"]).reset_index(drop=True)

    steps = pd.DataFrame(steps, columns=["step", "rows"])
    if verbose:
        print(steps.to_string(index=False))
    return df, steps


TABLE_PATH = ds.PROCESSED / "model_table.parquet"


def save_model_table():
    """Build the modeling table and save it to data/processed/ (parquet + CSV copy for browsing).

    build_model_table() stays the source of truth: re-run this after any change to it or to the
    processed inputs, so the saved file never drifts from the code.
    """
    df, steps = build_model_table(verbose=True)
    df.to_parquet(TABLE_PATH, index=False)
    df.to_csv(TABLE_PATH.with_suffix(".csv"), index=False)
    steps.to_csv(TABLE_PATH.with_name("model_table_filter_steps.csv"), index=False)
    print(f"saved {len(df):,} rows x {df.shape[1]} columns -> {TABLE_PATH.name} (+ .csv)")
    return df


def load_model_table():
    """Read the saved modeling table (fast; no rebuild). Call save_model_table() first if missing."""
    return pd.read_parquet(TABLE_PATH) if TABLE_PATH.exists() else save_model_table()


def _county_columns():
    """Static per-county predictors: host plants and county features."""
    feat = ds.load_county_features()[["fips", "metro", "census_division", "has_interstate",
                                      "rail_km_per_km2", "interstate_km_per_km2", "rucc_2023",
                                      "ecoregion_l3", "pop_density_km2"]]
    fia = pd.read_parquet(ds.PROCESSED / "county_hosts.parquet")[["fips", "tree_of_heaven_trees"]]

    # Tree of Heaven sighting rate from TRAINING years only, so test years can't leak in.
    toh = ds.fetch_gbif_county_counts(YEARS, taxon_key=ds.GBIF_TOH_TAXON_KEY)
    eff = pd.read_parquet(ds.PROCESSED / "county_year_effort.parquet")
    tr = lambda d: d[d["year"] <= max(TRAIN_YEARS)].groupby("fips")
    rate = pd.concat([tr(toh)["n_obs"].sum().rename("toh_obs"), tr(eff)["all_obs"].sum()], axis=1)
    rate["toh_rate_train"] = 1e4 * rate["toh_obs"].fillna(0) / rate["all_obs"]

    out = (feat.merge(fia, on="fips", how="left")
               .merge(rate[["toh_rate_train"]].reset_index(), on="fips", how="left"))
    out["fia_missing"] = out["tree_of_heaven_trees"].isna().astype(int)
    out["fia_toh_present"] = (out["tree_of_heaven_trees"].fillna(0) > 0).astype(int)
    out["toh_rate_train"] = out["toh_rate_train"].fillna(0)
    out["log_toh_rate_train"] = np.log1p(out["toh_rate_train"])  # rate is very right-skewed
    out["is_metro"] = (out["metro"] == "Metro").astype(int)
    out["has_interstate"] = out["has_interstate"].astype(int)
    for div in ["Middle Atlantic", "New England", "East North Central", "East South Central",
                "West North Central", "West South Central"]:
        out["div_" + div.lower().replace(" ", "_")] = (out["census_division"] == div).astype(int)
    return out.drop(columns=["tree_of_heaven_trees"])


def split(df):
    """Train/test split by year; asserts no overlap."""
    train, test = df[df["split"] == "train"].reset_index(drop=True), df[df["split"] == "test"].reset_index(drop=True)
    assert set(train["year"]).isdisjoint(test["year"])
    return train, test


def scores(y_true_log, y_pred_log, count_true=None, count_pred=None):
    """Metrics on the log1p scale (comparable with notebook 03), plus count-scale MAE if given."""
    out = {"test_RMSE": mean_squared_error(y_true_log, y_pred_log) ** .5,
           "test_MAE": mean_absolute_error(y_true_log, y_pred_log),
           "test_R2": r2_score(y_true_log, y_pred_log)}
    if count_true is not None:
        out["test_MAE_count"] = mean_absolute_error(count_true, count_pred)
    return out


def log_experiment(notebook, model, feature_set, metrics, n_train, n_test, notes=""):
    """Append one run to experiments/experiment_log.csv.

    Re-running a notebook replaces its earlier row for the same (notebook, model, feature_set),
    so the log holds each experiment's latest result rather than duplicates.
    """
    row = {"date": date.today().isoformat(), "notebook": notebook, "model": model,
           "feature_set": feature_set, "train_years": f"{min(TRAIN_YEARS)}-{max(TRAIN_YEARS)}",
           "test_years": f"{min(TEST_YEARS)}-{max(TEST_YEARS)}", "n_train": n_train, "n_test": n_test,
           **{k: round(float(v), 4) for k, v in metrics.items()}, "notes": notes}
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log = pd.read_csv(LOG_PATH) if LOG_PATH.exists() else pd.DataFrame()
    if len(log):
        same = (log["notebook"] == notebook) & (log["model"] == model) & (log["feature_set"] == feature_set)
        log = log[~same]
    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    log.to_csv(LOG_PATH, index=False)
    return row


def read_log():
    return pd.read_csv(LOG_PATH) if LOG_PATH.exists() else pd.DataFrame()


if __name__ == "__main__":
    save_model_table()
