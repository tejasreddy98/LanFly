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

st.set_page_config(page_title="LanFly", page_icon="🦗", layout="wide")


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
    return cy, geo, log, coef


cy, geo, log, coef = load()

MAP_METRICS = {
    "Lanternfly sightings": ("slf_obs", "log"),
    "Sightings per 10k observations (effort-adjusted)": ("slf_per_10k", "log"),
    "Predicted sightings (Elastic Net B)": ("pred_count", "log"),
    "Prediction error (log scale, predicted − actual)": ("resid", "diverging"),
    "Observer effort (all iNaturalist observations)": ("all_obs", "log"),
    "Average annual temperature (°F)": ("tavg_annual", "linear"),
    "Coldest winter month low (°F)": ("tmin_winter", "linear"),
    "Annual precipitation (in)": ("precip_annual", "linear"),
}
cy["resid"] = cy["pred_log"] - cy["y"]

st.sidebar.title("🦗 LanFly")
page = st.sidebar.radio("Page", ["Map explorer", "County predictions", "Model comparison", "About"])
st.sidebar.caption("DS 6021 final project · Reddy, Vrouwenvelder, Safdari, Wagner, Witzel")

# ---------------------------------------------------------------- Map explorer
if page == "Map explorer":
    st.title("Where are spotted lanternflies being reported?")
    c1, c2 = st.columns([2, 3])
    year = c1.slider("Year", int(cy["year"].min()), int(cy["year"].max()), 2025)
    metric = c2.selectbox("Show", list(MAP_METRICS))
    col, scale = MAP_METRICS[metric]

    d = cy[cy["year"] == year].copy()
    if col in ("pred_count", "resid") and d[col].isna().all():
        st.info("Predictions exist only for modeled county-years (2015–2025, enough observer effort).")
    shown = d[col]
    color_args = {}
    if scale == "log":
        d["value"] = np.log10(shown + 1)
        ticks = [0, 1, 2, 3, 4, 5]
        color_args = dict(color_continuous_scale="YlOrRd", range_color=(0, max(1, d["value"].quantile(.995))))
        cbar = dict(tickvals=ticks, ticktext=[f"{10**t - 1:,.0f}" for t in ticks], title="")
    elif scale == "diverging":
        d["value"] = shown
        color_args = dict(color_continuous_scale="RdBu_r", range_color=(-2, 2))
        cbar = dict(title="")
    else:
        d["value"] = shown
        color_args = dict(color_continuous_scale="Viridis")
        cbar = dict(title="")

    fig = px.choropleth(d, geojson=geo, locations="fips", featureidkey="properties.fips", color="value",
                        hover_name="label",
                        hover_data={"fips": False, "value": False, "slf_obs": ":,", "all_obs": ":,",
                                    "pred_count": ":,.0f"},
                        labels={"slf_obs": "Sightings", "all_obs": "All observations",
                                "pred_count": "Predicted"},
                        **color_args)
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(height=620, margin=dict(l=0, r=0, t=0, b=0), coloraxis_colorbar=cbar)
    st.plotly_chart(fig, width="stretch")

    a, b, c = st.columns(3)
    a.metric("Counties with sightings", f"{(d['slf_obs'] > 0).sum():,}")
    b.metric("Total sightings", f"{int(d['slf_obs'].sum()):,}")
    c.metric("Observer effort (all obs.)", f"{int(d['all_obs'].sum()):,}")
    st.caption("Sightings are iNaturalist research-grade observations, a proxy for population that also "
               "depends on how many people are looking. Grey counties have no data for that year.")

# ---------------------------------------------------------- County predictions
elif page == "County predictions":
    st.title("Actual vs. predicted sightings by county")
    modeled = cy.dropna(subset=["pred_log"])
    top = (modeled.groupby("label")["slf_obs"].sum().sort_values(ascending=False).index.tolist())
    county = st.selectbox("County (sorted by total sightings)", top)
    d = cy[cy["label"] == county].sort_values("year")

    fig = go.Figure()
    fig.add_vrect(x0=2022.5, x1=2025.5, fillcolor="#B23A24", opacity=.08, line_width=0,
                  annotation_text="test years", annotation_position="top left")
    fig.add_trace(go.Scatter(x=d["year"], y=d["slf_obs"], mode="lines+markers", name="Actual",
                             line=dict(color="#1F2A30")))
    fig.add_trace(go.Scatter(x=d["year"], y=d["pred_count"], mode="lines+markers", name="Predicted (Elastic Net B)",
                             line=dict(color="#2D6A6E", dash="dash")))
    fig.update_layout(height=450, yaxis_title="Lanternfly sightings", xaxis=dict(dtick=1),
                      legend=dict(orientation="h", y=1.1), margin=dict(t=40))
    st.plotly_chart(fig, width="stretch")

    t = d[["year", "split", "slf_obs", "pred_count", "all_obs", "tavg_annual", "tmin_winter", "precip_annual"]]
    t = t.rename(columns={"split": "set", "slf_obs": "actual", "pred_count": "predicted",
                          "all_obs": "all observations", "tavg_annual": "avg temp °F",
                          "tmin_winter": "winter low °F", "precip_annual": "precip in"})
    st.dataframe(t.round(1), hide_index=True, width="stretch")
    st.caption("The model was trained on 2015–2022 only; 2023–2025 are out-of-sample predictions. "
               "Years with no prediction were outside the modeling filters (too little observer effort "
               "or incomplete climate data).")

# ------------------------------------------------------------ Model comparison
elif page == "Model comparison":
    st.title("How do the models compare?")
    st.write("Every model is trained on 2015–2022 and tested on 2023–2025, on the same rows. "
             "Lower RMSE and MAE are better; scores are on the log(1 + count) scale.")
    log["run"] = log["model"] + log["feature_set"].map(lambda f: "" if f == "-" else f" ({f})")
    order = log.sort_values("test_RMSE")
    fig = px.bar(order, x="test_RMSE", y="run", orientation="h", color="notebook",
                 labels={"test_RMSE": "Test RMSE (lower is better)", "run": ""}, text="test_RMSE")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(height=420, yaxis=dict(autorange="reversed"), margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(order[["run", "notebook", "test_RMSE", "test_MAE", "test_R2", "n_train", "n_test", "notes"]],
                 hide_index=True, width="stretch")

    st.subheader("What drives the best model?")
    c = coef.reindex(coef["coef_std"].abs().sort_values().index)
    fig = px.bar(c, x="coef_std", y="feature", orientation="h",
                 labels={"coef_std": "Standardized coefficient (log scale)", "feature": ""})
    fig.update_layout(height=360, margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("Coefficients are per one standard deviation of each feature. Temperature features are "
               "strongly correlated with each other, so read them together rather than one at a time.")

# ----------------------------------------------------------------------- About
else:
    st.title("About this project")
    st.markdown("""
**Question.** How many spotted lanternfly sightings will a county report in a given year?

**Data.** iNaturalist research-grade sightings (via GBIF), NOAA county climate, USFS forest
inventory and GBIF host-plant records, and county features (rural–urban codes, roads, rail).

**Unit.** One row per county per year, eastern US, 2014–2025.

**Validation.** Train on 2015–2022, test on 2023–2025, so every score is a true forecast.

Source code and notebooks: [github.com/tejasreddy98/LanFly](https://github.com/tejasreddy98/LanFly)
""")
