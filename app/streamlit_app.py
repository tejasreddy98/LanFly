# AI-GENERATED (Claude Code, 2026-11-20). Verification steps are recorded in docs/ai_usage.md.
"""LanFly dashboard: spotted lanternfly sightings, predictions and model comparison.

Run locally from the repo root:
    streamlit run app/streamlit_app.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP = Path(__file__).resolve().parent
DATA = APP / "data"
LOG = APP.parent / "experiments" / "experiment_log.csv"

st.set_page_config(page_title="Spotted Lanternfly Population Model", layout="wide")


@st.cache_data
def load():
    cy = pd.read_parquet(DATA / "county_year.parquet")
    cy["label"] = cy["county"] + ", " + cy["state"]
    with open(DATA / "counties.geojson") as f:
        geo = json.load(f)
    in_box = {feat["properties"]["fips"] for feat in geo["features"]}
    cy = cy[cy["fips"].isin(in_box)]
    log = pd.read_csv(LOG)
    coef = pd.read_csv(DATA / "elasticnet_b_coefficients.csv")
    hosts = pd.read_parquet(DATA / "county_hosts.parquet")
    preds = pd.read_parquet(DATA / "model_predictions.parquet")
    return cy, geo, log, coef, hosts, preds


cy, geo, log, coef, hosts, preds = load()

MAP_METRICS = {
    "Reported sightings (count)": ("slf_obs", "log"),
    "Sightings per 10,000 observations (effort-adjusted rate)": ("slf_per_10k", "log"),
    "Predicted sightings (Elastic Net, feature set B)": ("pred_count", "log"),
    "Prediction residual, log(1 + count) scale (predicted − observed)": ("resid", "diverging"),
    "Observer effort (total iNaturalist observations)": ("all_obs", "log"),
    "Mean annual temperature (°F)": ("tavg_annual", "linear"),
    "Minimum winter temperature (°F)": ("tmin_winter", "linear"),
    "Annual precipitation (in)": ("precip_annual", "linear"),
}
cy["resid"] = cy["pred_log"] - cy["y"]

st.sidebar.title("Spotted Lanternfly Population Model")
page = st.sidebar.radio("Section", ["Spatial Distribution", "County-Level Predictions", "Model Evaluation", "Project Overview"])
st.sidebar.caption("DS 6021 Final Project, University of Virginia\n\nT. Reddy, L. Vrouwenvelder, N. Safdari, M. Wagner, A. Witzel")

# ---------------------------------------------------------------- Map helpers
TICK_COUNTS = [0, 1, 10, 100, 1000, 10000]
LOG_TICKS = [float(np.log10(v + 1)) for v in TICK_COUNTS]
TICK_TEXT = [f"{v:,}" for v in TICK_COUNTS]


def county_map(d, value, scale, hover, height=560, vmax=None, title=None):
    """Choropleth of one column. scale: 'log' (log10(1+x) colors, count tick labels), 'diverging', 'linear'."""
    d = d.copy()
    if scale == "log":
        d["_v"] = np.log10(d[value] + 1)
        hi = vmax if vmax is not None else max(1, d["_v"].quantile(.995))
        args = dict(color_continuous_scale="YlOrRd", range_color=(0, hi))
        cbar = dict(tickvals=LOG_TICKS, ticktext=TICK_TEXT, title="")
    elif scale == "diverging":
        d["_v"] = d[value]
        args = dict(color_continuous_scale="RdBu_r", range_color=(-2, 2))
        cbar = dict(title="")
    else:
        d["_v"] = d[value]
        # Percentile limits keep one extreme county from washing out the rest of the map.
        args = dict(color_continuous_scale="Viridis",
                    range_color=(d["_v"].quantile(.01), d["_v"].quantile(.99)))
        cbar = dict(title="")
    fig = px.choropleth(d, geojson=geo, locations="fips", featureidkey="properties.fips", color="_v",
                        hover_name="label", hover_data={"fips": False, "_v": False, **hover},
                        labels={**HOVER_LABELS, "toh_obs": "Tree of Heaven sightings"}, **args)
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=30 if title else 0, b=0),
                      coloraxis_colorbar=cbar, title=title)
    return fig


HOVER = {"slf_obs": ":,", "all_obs": ":,", "pred_count": ":,.0f"}
HOVER_LABELS = {"slf_obs": "Observed", "all_obs": "Observer effort", "pred_count": "Predicted"}

# ---------------------------------------------------------------- Spatial distribution
if page == "Spatial Distribution":
    st.title("Spatial Distribution of Reported Sightings")
    tab1, tab2, tab3 = st.tabs(["Single variable", "Observed vs. predicted", "Host plants"])

    with tab1:
        c1, c2 = st.columns([2, 3])
        year = c1.slider("Year", int(cy["year"].min()), int(cy["year"].max()), 2025, key="y1")
        metric = c2.selectbox("Variable", list(MAP_METRICS))
        col, scale = MAP_METRICS[metric]
        d = cy[cy["year"] == year]
        if col in ("pred_count", "resid") and d[col].isna().all():
            st.info("Predictions are available only for county-years included in the modeling sample "
                    "(2015–2025, with sufficient observer effort and complete climate records).")
        fig = county_map(d, col, scale, HOVER, height=620)
        st.plotly_chart(fig, width="stretch")
        a, b, c = st.columns(3)
        a.metric("Counties with ≥1 sighting", f"{(d['slf_obs'] > 0).sum():,}")
        b.metric("Total reported sightings", f"{int(d['slf_obs'].sum()):,}")
        c.metric("Total observer effort", f"{int(d['all_obs'].sum()):,}")
        st.caption("Sightings are research-grade iNaturalist observations, used here as a proxy for population "
                   "abundance; reported counts also depend on observer effort. Counties without data for the "
                   "selected year are not shaded.")

    with tab2:
        modeled_years = sorted(cy.dropna(subset=["pred_log"])["year"].unique())
        year2 = st.select_slider("Year", options=modeled_years, value=modeled_years[-1], key="y2")
        d = cy[(cy["year"] == year2)].dropna(subset=["pred_count"])
        vmax = max(1, np.log10(d[["slf_obs", "pred_count"]].to_numpy().max() + 1))
        m1, m2 = st.columns(2)
        m1.plotly_chart(county_map(d, "slf_obs", "log", HOVER, height=430, vmax=vmax, title="Observed"),
                        width="stretch")
        m2.plotly_chart(county_map(d, "pred_count", "log", HOVER, height=430, vmax=vmax,
                                   title="Predicted (Elastic Net, feature set B)"), width="stretch")

        s1, s2 = st.columns([3, 2])
        lim = np.log10(max(d["slf_obs"].max(), d["pred_count"].max()) + 1) + .1
        sc = px.scatter(d.assign(obs=np.log10(d["slf_obs"] + 1), pred=np.log10(d["pred_count"] + 1)),
                        x="obs", y="pred", hover_name="label", opacity=.6,
                        hover_data={"obs": False, "pred": False, "slf_obs": ":,", "pred_count": ":,.0f"},
                        labels={"obs": "Observed sightings", "pred": "Predicted sightings", **HOVER_LABELS})
        sc.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim, line=dict(dash="dash", color="grey"))
        ticks = dict(tickvals=LOG_TICKS, ticktext=TICK_TEXT, range=[-.05, lim])
        sc.update_layout(height=380, xaxis=ticks, yaxis=ticks, margin=dict(t=30), title="County-level agreement")
        s1.plotly_chart(sc, width="stretch")
        err = d["pred_log"] - d["y"]
        s2.metric("Counties in modeling sample", f"{len(d):,}")
        s2.metric("RMSE, log(1 + count)", f"{np.sqrt((err ** 2).mean()):.3f}")
        s2.metric("Sample", "Test (out-of-sample)" if year2 >= 2023 else "Training (in-sample)")
        st.caption("Both maps share one color scale. Points on the dashed line are predicted exactly; points "
                   "below it are under-predicted. Only county-years in the modeling sample are shown.")

    with tab3:
        HOSTS = {
            "Tree of Heaven, share of forest trees (FIA)": ("tree_of_heaven_share", "pct"),
            "Tree of Heaven, iNaturalist sightings per 10,000 observations": ("toh_per_10k", "log"),
            "Black walnut, share of forest trees (FIA)": ("black_walnut_share", "pct"),
            "Maples, share of forest trees (FIA)": ("maple_share", "pct"),
            "Willows, share of forest trees (FIA)": ("willow_share", "pct"),
        }
        hsel = st.selectbox("Host variable", list(HOSTS))
        hcol, hscale = HOSTS[hsel]
        labels = cy[["fips", "label"]].drop_duplicates()
        h = hosts.merge(labels, on="fips")
        cum = cy.groupby("fips")[["slf_obs", "all_obs"]].sum()
        h = h.merge((1e4 * cum["slf_obs"] / cum["all_obs"]).rename("slf_rate").reset_index(), on="fips", how="left")
        if hscale == "pct":
            h["_pct"] = 100 * h[hcol]
            fig = county_map(h, "_pct", "linear", {"_pct": ":.2f"}, height=560)
            fig.update_traces(hovertemplate="<b>%{hovertext}</b><br>%{z:.2f}%<extra></extra>")
            fig.update_layout(coloraxis_colorbar=dict(title="%"))
        else:
            fig = county_map(h, hcol, "log", {"toh_obs": ":,"}, height=560)
        st.plotly_chart(fig, width="stretch")

        valid = h[[hcol, "slf_rate"]].dropna()
        rho = valid[hcol].rank().corr(valid["slf_rate"].rank())
        st.metric(f"Spearman correlation with lanternfly sightings per 10,000 observations (2014–2025), "
                  f"n = {len(valid):,}", f"{rho:.2f}")
        st.caption("Tree of Heaven (*Ailanthus altissima*) is the preferred host of the spotted lanternfly; "
                   "black walnut, maples and willows are secondary hosts. FIA shares are estimated from "
                   "forest-inventory plots and are missing for counties without sampled forest land. Host "
                   "variables are static (one value per county).")

# ---------------------------------------------------------- County predictions
elif page == "County-Level Predictions":
    st.title("Observed and Predicted Sightings by County")
    modeled = cy.dropna(subset=["pred_log"])
    totals = modeled.groupby("label")["slf_obs"].sum()
    options = sorted(totals.index)
    c1, _ = st.columns([1, 2])
    county = c1.selectbox("County", options, index=options.index(totals.idxmax()))
    model_rmse = preds.groupby("model")["test_rmse"].first().sort_values()
    chosen = st.multiselect(
        "Models (ordered by test RMSE)", list(model_rmse.index),
        default=["Elastic Net (B)", "Persistence baseline", "Negative binomial (B)"],
        format_func=lambda n: f"{n}  ·  RMSE {model_rmse[n]:.3f}")
    d = cy[cy["label"] == county].sort_values("year")
    fips = d["fips"].iloc[0]
    p = preds[(preds["fips"] == fips) & preds["model"].isin(chosen)]

    fig = go.Figure()
    fig.add_vrect(x0=2022.5, x1=2025.5, fillcolor="#B23A24", opacity=.08, line_width=0,
                  annotation_text="Test period (2023–2025)", annotation_position="top left")
    fig.add_trace(go.Scatter(x=d["year"], y=d["slf_obs"], mode="lines+markers", name="Observed",
                             line=dict(color="#1F2A30", width=3)))
    palette = px.colors.qualitative.Safe
    for i, name in enumerate(chosen):
        pm = p[p["model"] == name].sort_values("year")
        fig.add_trace(go.Scatter(x=pm["year"], y=pm["pred_count"], mode="lines+markers", name=name,
                                 line=dict(color=palette[i % len(palette)], dash="dash")))
    log_y = st.toggle("Logarithmic y-axis", value=False)
    fig.update_layout(height=470, yaxis_title="Reported sightings", xaxis_title="Year", xaxis=dict(dtick=1),
                      legend=dict(orientation="h", y=1.12), margin=dict(t=50))
    if log_y:
        fig.update_yaxes(type="log")
    st.plotly_chart(fig, width="stretch")

    wide = p.pivot_table(index="year", columns="model", values="pred_count").reindex(columns=chosen)
    t = (d.set_index("year")[["split", "slf_obs", "all_obs", "tavg_annual", "tmin_winter", "precip_annual"]]
           .join(wide).reset_index())
    t = t.rename(columns={"year": "Year", "split": "Sample", "slf_obs": "Observed",
                          "all_obs": "Observer effort", "tavg_annual": "Mean temp. (°F)",
                          "tmin_winter": "Min. winter temp. (°F)", "precip_annual": "Precip. (in)"})
    st.dataframe(t.round(1), hide_index=True, width="stretch")
    st.caption("All models were fitted on 2015–2022; predictions for 2023–2025 are out-of-sample. Predictions "
               "are back-transformed from the log(1 + count) scale on which the models are scored; for the "
               "negative binomial this is the simulated expected log(1 + count), not the predicted mean. The "
               "persistence baseline predicts the previous year's count. Years without a prediction fall "
               "outside the modeling sample (insufficient observer effort or incomplete climate records).")

# ------------------------------------------------------------ Model comparison
elif page == "Model Evaluation":
    st.title("Model Evaluation")
    st.write("All models are fitted on 2015–2022 and evaluated on a held-out test period (2023–2025) "
             "using identical observations. Error metrics are computed on the log(1 + count) scale; "
             "lower RMSE and MAE indicate better predictive accuracy.")
    log["run"] = log["model"] + log["feature_set"].map(lambda f: "" if f == "-" else f" ({f})")
    order = log.sort_values("test_RMSE")
    fig = px.bar(order, x="test_RMSE", y="run", orientation="h", color="notebook",
                 labels={"test_RMSE": "Test RMSE", "run": "", "notebook": "Notebook"}, text="test_RMSE")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(height=420, yaxis=dict(autorange="reversed"), margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(order[["run", "notebook", "test_RMSE", "test_MAE", "test_R2", "n_train", "n_test", "notes"]].rename(columns={
                     "run": "Model", "notebook": "Notebook", "test_RMSE": "RMSE", "test_MAE": "MAE",
                     "test_R2": "R²", "n_train": "n (train)", "n_test": "n (test)", "notes": "Notes"}),
                 hide_index=True, width="stretch")

    st.subheader("Standardized Coefficients of the Best-Performing Model")
    names = {"log_effort": "Observer effort (log)", "tavg_annual": "Mean annual temperature",
             "tmin_winter": "Minimum winter temperature", "tmax_summer": "Mean summer maximum temperature",
             "precip_annual": "Annual precipitation", "lat": "Latitude", "lon": "Longitude",
             "lag_log_count": "Previous-year sightings (log)"}
    c = coef.assign(feature=coef["feature"].map(names).fillna(coef["feature"]))
    c = c.reindex(c["coef_std"].abs().sort_values().index)
    fig = px.bar(c, x="coef_std", y="feature", orientation="h",
                 labels={"coef_std": "Standardized coefficient (log scale)", "feature": ""})
    fig.update_layout(height=360, margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("Coefficients correspond to a one-standard-deviation change in each predictor. The temperature "
               "predictors are highly collinear, so their individual coefficients should be interpreted jointly.")

# ----------------------------------------------------------------------- About
else:
    st.title("Project Overview")
    st.markdown("""
**Research question.** Can the annual number of reported spotted lanternfly (*Lycorma delicatula*)
sightings in a county be predicted from observer effort, climate, host-plant availability and
county characteristics?

**Data sources.** Research-grade iNaturalist observations (accessed via GBIF); NOAA nClimDiv county
climate series; USDA Forest Service FIA inventory and GBIF records of host plants; USDA ERS
Rural–Urban Continuum Codes; EPA Level III ecoregions; and Census TIGER road and rail networks.

**Unit of analysis.** County-year, eastern United States, 2014–2025.

**Validation design.** Temporal holdout: models are fitted on 2015–2022 and evaluated on 2023–2025,
so all reported scores reflect out-of-sample forecasting performance.

Source code and notebooks: [github.com/tejasreddy98/LanFly](https://github.com/tejasreddy98/LanFly)
""")
