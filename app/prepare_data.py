# AI-GENERATED (Claude Code, 2026-11-20). Verification steps are recorded in docs/ai_usage.md.
"""Build the small data files the Streamlit app reads (app/data/).

Run from the repo root after the notebooks' processed data exists:
    python app/prepare_data.py

The deployed app never touches data/raw or the network: it only reads the files
written here, so they are committed to git.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import data_sources as ds  # noqa: E402
import modeling as m  # noqa: E402

OUT = ROOT / "app" / "data"
OUT.mkdir(parents=True, exist_ok=True)

# 1. Shared modeling table and the best model so far (Elastic Net, feature set B),
#    fitted exactly as in notebook 03: same alphas, l1 ratios and expanding-window CV.
df, _ = m.build_model_table(verbose=False)
train, test = m.split(df)
feats = m.FEATURE_SETS["B"]
cv = [(np.flatnonzero(train["year"] < v), np.flatnonzero(train["year"] == v)) for v in range(2019, 2023)]
model = make_pipeline(StandardScaler(), ElasticNetCV(alphas=np.logspace(-6, 3, 80), l1_ratio=[.1, .5, .9],
                                                     cv=cv, max_iter=50_000)).fit(train[feats], train["y"])
df["pred_log"] = model.predict(df[feats])
df["pred_count"] = np.expm1(df["pred_log"]).clip(lower=0)
check = m.scores(test["y"], model.predict(test[feats]))
print("Elastic Net B test scores:", {k: round(v, 4) for k, v in check.items()})

coef = pd.DataFrame({"feature": feats, "coef_std": model[-1].coef_})
coef.to_csv(OUT / "elasticnet_b_coefficients.csv", index=False)

# 1b. Predictions from several models for the county page, all on the log1p scale.
#     Settings mirror notebooks 03 (Ridge/Lasso/Elastic Net) and 04 (negative binomial).
import statsmodels.api as sm  # noqa: E402
from sklearn.linear_model import LassoCV, RidgeCV  # noqa: E402

ALPHAS = np.logspace(-6, 3, 80)
sk_models = {
    "Elastic Net (B)": (model, "B"),
    "Elastic Net (A)": (make_pipeline(StandardScaler(), ElasticNetCV(alphas=ALPHAS, l1_ratio=[.1, .5, .9],
                                                                     cv=cv, max_iter=50_000)), "A"),
    "Ridge (B)": (make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS, cv=cv)), "B"),
    "Lasso (B)": (make_pipeline(StandardScaler(), LassoCV(alphas=ALPHAS, cv=cv, max_iter=50_000)), "B"),
}
preds = {"Persistence baseline": df["lag_log_count"].to_numpy()}
for name, (mdl, fs) in sk_models.items():
    f = m.FEATURE_SETS[fs]
    if mdl is not model:
        mdl.fit(train[f], train["y"])
    preds[name] = mdl.predict(df[f])


def nb_elog1p(fs, draws=4000, seed=6021):
    """NB2 fitted as in notebook 04; returns simulated E[log1p(Y)] for every row of df."""
    f = m.FEATURE_SETS[fs]
    sc = StandardScaler().fit(train[f])
    Xtr, Xall = sm.add_constant(sc.transform(train[f])), sm.add_constant(sc.transform(df[f]))
    ytr = train["slf_obs"].to_numpy()
    pois = sm.GLM(ytr, Xtr, family=sm.families.Poisson()).fit()
    nbm = sm.NegativeBinomial(ytr, Xtr, loglike_method="nb2")
    res = nbm.fit(start_params=np.append(np.asarray(pois.params), 1.0), method="newton", maxiter=200, disp=0)
    if not res.mle_retvals["converged"]:
        res = nbm.fit(start_params=np.asarray(res.params), method="bfgs", maxiter=5000, disp=0)
    p = np.asarray(res.params)
    mu, size = np.exp(Xall @ p[:-1]), 1 / p[-1]
    rng = np.random.default_rng(seed)
    sims = rng.negative_binomial(size, (size / (size + mu))[:, None], size=(len(mu), draws))
    return np.log1p(sims).mean(axis=1)


for fs in ["B", "C"]:
    preds[f"Negative binomial ({fs})"] = nb_elog1p(fs)

long = []
is_test = (df["split"] == "test").to_numpy()
for name, p in preds.items():
    rmse = float(np.sqrt(((p[is_test] - df.loc[is_test, "y"]) ** 2).mean()))
    print(f"  {name:<24} test RMSE {rmse:.4f}")
    long.append(pd.DataFrame({"fips": df["fips"], "year": df["year"], "model": name, "pred_log": p,
                              "pred_count": np.expm1(p).clip(min=0), "test_rmse": rmse}))
pd.concat(long).to_parquet(OUT / "model_predictions.parquet", index=False)

# 2. County-year table for the maps: every county-year in the region box (not just modeled rows).
cy = pd.read_parquet(ds.PROCESSED / "county_year_effort.parquet")
cy = cy[cy["year"].isin(m.YEARS)]
counties = ds.load_counties()
keep_cols = ["fips", "year", "split", "y", "pred_log", "pred_count", "lag_log_count",
             "tavg_annual", "tmin_winter", "tmax_summer", "precip_annual",
             "toh_rate_train", "metro", "census_division"]
app_cy = cy.merge(df[keep_cols], on=["fips", "year"], how="left")  # cy already has state, county, slf_per_10k
app_cy.to_parquet(OUT / "county_year.parquet", index=False)

# 3. Simplified county outlines for the region box only (keeps the file small).
geo = counties.copy()
cent = geo.to_crs(5070).geometry.centroid.to_crs(4326)
geo = geo[cent.x.between(*m.REGION["lon"]) & cent.y.between(*m.REGION["lat"])]
geo = geo.to_crs(5070)
geo["geometry"] = geo.geometry.simplify(1500)
geo = geo.to_crs(4326)[["fips", "geometry"]]
geo.to_file(OUT / "counties.geojson", driver="GeoJSON", COORDINATE_PRECISION=4)

# 4. Host plants per county (static): FIA forest inventory shares and GBIF Tree of Heaven sightings.
hosts = pd.read_parquet(ds.PROCESSED / "county_hosts.parquet")[
    ["fips", "tree_of_heaven_share", "black_walnut_share", "maple_share", "willow_share",
     "tree_of_heaven_trees", "toh_obs", "toh_per_10k"]]
hosts = hosts[hosts["fips"].isin(geo["fips"])]
hosts.to_parquet(OUT / "county_hosts.parquet", index=False)

for p in sorted(OUT.iterdir()):
    print(f"{p.name}: {p.stat().st_size / 1e3:,.0f} KB")
