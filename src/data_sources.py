# AI-GENERATED (Claude Code, 2026-10-27). Verification steps are recorded in docs/ai_usage.md.
"""Download and load the raw data sources for the lanternfly project.

Every download is cached in data/raw/, so re-running only hits the network
when a file is missing (or refresh=True).
"""

from pathlib import Path
import time
import unicodedata

import geopandas as gpd
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# iNaturalist: spotted lanternfly (Lycorma delicatula) observations
# ---------------------------------------------------------------------------

INAT_API = "https://api.inaturalist.org/v1/observations"
SLF_TAXON_ID = 324726  # Lycorma delicatula
US_PLACE_ID = 1        # United States


def fetch_inat(refresh=False, quality_grade="research", per_page=200, pause=1.0):
    """Return all US spotted lanternfly observations as a DataFrame.

    Pages through the API with id_above (the API caps plain paging at 10k
    records) and sleeps between requests to respect iNaturalist's rate limit.
    Each page is appended to a .partial file, so an interrupted download
    resumes where it stopped instead of starting over.
    """
    path = RAW / f"inat_slf_{quality_grade}.csv"
    partial = path.with_suffix(".partial.csv")
    if path.exists() and not refresh:
        return pd.read_csv(path, parse_dates=["observed_on"])
    if refresh:
        partial.unlink(missing_ok=True)

    RAW.mkdir(parents=True, exist_ok=True)
    last_id, n = 0, 0
    if partial.exists():
        done = pd.read_csv(partial, usecols=["id"])["id"]
        last_id, n = int(done.max()), len(done)
        print(f"Resuming after {n:,} saved observations")

    with requests.Session() as s:
        while True:
            params = {
                "taxon_id": SLF_TAXON_ID,
                "place_id": US_PLACE_ID,
                "quality_grade": quality_grade,
                "per_page": per_page,
                "order_by": "id",
                "order": "asc",
                "id_above": last_id,
            }
            results = _get_with_retry(s, INAT_API, params)["results"]
            if not results:
                break
            page = pd.DataFrame([_flatten_obs(o) for o in results])
            page.to_csv(partial, mode="a", header=not partial.exists(), index=False)
            last_id = results[-1]["id"]
            n += len(results)
            print(f"{n:,} observations", flush=True)
            time.sleep(pause)

    partial.rename(path)
    return pd.read_csv(path, parse_dates=["observed_on"])


def _get_with_retry(session, url, params, tries=6):
    """GET JSON, backing off on rate limits (429), server errors, and timeouts."""
    for attempt in range(tries):
        try:
            r = session.get(url, params=params, timeout=60)
            if r.status_code != 429 and r.status_code < 500:
                r.raise_for_status()
                return r.json()
            print(f"HTTP {r.status_code}; retrying", flush=True)
        except (requests.ConnectionError, requests.Timeout) as e:
            print(f"{type(e).__name__}; retrying", flush=True)
        time.sleep(min(2 ** attempt * 5, 120))
    raise RuntimeError(f"iNaturalist request failed after {tries} tries: {params}")


def _flatten_obs(o):
    """Keep only the fields we need; no usernames or free-text place names."""
    lat = lon = None
    if o.get("location"):
        lat, lon = (float(v) for v in o["location"].split(","))
    return {
        "id": o["id"],
        "observed_on": o.get("observed_on"),
        "latitude": lat,
        "longitude": lon,
        "positional_accuracy_m": o.get("positional_accuracy"),
        "obscured": o.get("obscured"),
        "quality_grade": o.get("quality_grade"),
        "user_id": (o.get("user") or {}).get("id"),  # anonymous id, useful for observer-effort features
    }


# ---------------------------------------------------------------------------
# US county boundaries (Census cartographic boundary file)
# ---------------------------------------------------------------------------
COUNTY_URL = "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_20m.zip"


def load_counties(refresh=False):
    """Return US county polygons with a 5-digit FIPS column."""
    path = RAW / "cb_2020_us_county_20m.zip"
    _download(COUNTY_URL, path, refresh)
    counties = gpd.read_file(path)
    counties = counties.rename(columns={"GEOID": "fips", "NAME": "county", "STUSPS": "state"})
    return counties[["fips", "state", "county", "ALAND", "geometry"]]


def assign_county(obs, counties):
    """Add the county FIPS each observation falls in (NaN if outside every county)."""
    pts = gpd.GeoDataFrame(
        obs,
        geometry=gpd.points_from_xy(obs["longitude"], obs["latitude"]),
        crs="EPSG:4326",
    ).to_crs(counties.crs)
    joined = gpd.sjoin(pts, counties[["fips", "state", "county", "geometry"]],
                       how="left", predicate="within")
    return pd.DataFrame(joined.drop(columns=["geometry", "index_right"]))


# ---------------------------------------------------------------------------
# NOAA nClimDiv: monthly county climate (PRISM-like gridded station averages)
# ---------------------------------------------------------------------------

CLIMDIV = "https://www.ncei.noaa.gov/pub/data/cirs/climdiv"
CLIMATE_VARS = {"tmpccy": "tavg_f", "tmaxcy": "tmax_f", "tmincy": "tmin_f", "pcpncy": "precip_in"}
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

# nClimDiv numbers states alphabetically (01-48 for CONUS), not by FIPS.
NCDC_TO_FIPS = {
    1: "01", 2: "04", 3: "05", 4: "06", 5: "08", 6: "09", 7: "10", 8: "12", 9: "13",
    10: "16", 11: "17", 12: "18", 13: "19", 14: "20", 15: "21", 16: "22", 17: "23",
    18: "24", 19: "25", 20: "26", 21: "27", 22: "28", 23: "29", 24: "30", 25: "31",
    26: "32", 27: "33", 28: "34", 29: "35", 30: "36", 31: "37", 32: "38", 33: "39",
    34: "40", 35: "41", 36: "42", 37: "44", 38: "45", 39: "46", 40: "47", 41: "48",
    42: "49", 43: "50", 44: "51", 45: "53", 46: "54", 47: "55", 48: "56",
}


def _latest_climdiv_version():
    return requests.get(f"{CLIMDIV}/procdate.txt", timeout=30).text.strip()


def load_climate(start_year=2010, refresh=False):
    """Return county-month climate in long form: fips, year, month, tavg_f, tmax_f, tmin_f, precip_in."""
    version = _latest_climdiv_version()
    frames = []
    for code, name in CLIMATE_VARS.items():
        path = RAW / f"climdiv-{code}-v1.0.0-{version}"
        _download(f"{CLIMDIV}/climdiv-{code}-v1.0.0-{version}", path, refresh)
        frames.append(_parse_climdiv(path, name, start_year))
    climate = frames[0]
    for f in frames[1:]:
        climate = climate.merge(f, on=["fips", "year", "month"], how="outer")
    return climate.sort_values(["fips", "year", "month"]).reset_index(drop=True)


def _parse_climdiv(path, value_name, start_year):
    widths = [2, 3, 2, 4] + [7] * 12
    df = pd.read_fwf(path, widths=widths, header=None,
                     names=["ncdc_state", "county_fips", "element", "year"] + MONTHS,
                     dtype={"county_fips": str})
    df = df[df["year"] >= start_year]
    df["fips"] = df["ncdc_state"].map(NCDC_TO_FIPS) + df["county_fips"].str.zfill(3)
    long = df.melt(id_vars=["fips", "year"], value_vars=MONTHS, var_name="month", value_name=value_name)
    long["month"] = long["month"].map({m: i + 1 for i, m in enumerate(MONTHS)})
    # Missing-value sentinels: the readme says -99.99 / -9.99, but the files
    # actually use -99.90 for temperature. Real monthly values never get that low
    # (precipitation is never negative), so a threshold is safe.
    floor = 0 if value_name == "precip_in" else -99
    long.loc[long[value_name] < floor, value_name] = pd.NA
    return long.dropna(subset=["fips"])


def _download(url, path, refresh=False):
    if path.exists() and not refresh:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


# ---------------------------------------------------------------------------
# GBIF: iNaturalist observation counts per county-year (observer effort)
# ---------------------------------------------------------------------------
# GBIF mirrors iNaturalist research-grade observations, but only openly
# licensed ones (~73% of SLF records). Use GBIF for BOTH effort and SLF counts
# when computing ratios so numerator and denominator come from the same subset.

GBIF = "https://api.gbif.org/v1"
GBIF_INAT_DATASET = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"  # iNaturalist research-grade
GBIF_SLF_TAXON_KEY = 5157899  # Lycorma delicatula (checked via /species/match)


def fetch_gbif_county_counts(years, taxon_key=None, refresh=False):
    """Return iNaturalist observation counts per county-year from GBIF.

    taxon_key=None counts all species (observer effort); pass GBIF_SLF_TAXON_KEY
    for lanternfly only. Columns: fips, year, n_obs.
    """
    label = "all" if taxon_key is None else f"taxon{taxon_key}"
    path = RAW / f"gbif_inat_county_counts_{label}.csv"
    cached = pd.read_csv(path, dtype={"fips": str}) if path.exists() and not refresh else None
    todo = [y for y in years if cached is None or y not in set(cached["year"])]

    frames = [] if cached is None else [cached]
    gadm = load_gadm_fips()
    with requests.Session() as s:
        for year in todo:
            params = {"datasetKey": GBIF_INAT_DATASET, "country": "US", "year": year,
                      "limit": 0, "facet": "gadmLevel2Gid", "facetLimit": 5000}
            if taxon_key is not None:
                params["taxonKey"] = taxon_key
            facets = _get_with_retry(s, f"{GBIF}/occurrence/search", params)["facets"]
            counts = pd.DataFrame(facets[0]["counts"] if facets else [], columns=["name", "count"])
            counts = counts.rename(columns={"name": "gadm_gid", "count": "n_obs"}).merge(gadm, on="gadm_gid")
            counts["year"] = year
            frames.append(counts[["fips", "year", "n_obs"]])
            print(f"{label} {year}: {counts['n_obs'].sum():,} observations", flush=True)
            time.sleep(0.5)

    out = pd.concat(frames, ignore_index=True).sort_values(["fips", "year"])
    out.to_csv(path, index=False)
    return out[out["year"].isin(years)].reset_index(drop=True)


def load_gadm_fips(refresh=False):
    """Map GBIF's GADM county IDs (e.g. USA.39.2_1) to Census FIPS by state + name."""
    path = RAW / "gadm_to_fips.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, dtype={"fips": str})

    rows, offset = [], 0
    with requests.Session() as s:
        while True:
            page = _get_with_retry(s, f"{GBIF}/geocode/gadm/search",
                                   {"gadmLevel": 2, "gadmGid": "USA", "limit": 1000, "offset": offset})
            for r in page["results"]:
                state = next(h["name"] for h in r["higherRegions"] if h["id"] != "USA")
                # GADM labels some independent cities (e.g. "Fairfax City") as counties.
                is_city = "Independent City" in r["englishType"] or r["name"].endswith(" City")
                rows.append({"gadm_gid": r["id"], "state_name": state, "name": r["name"], "is_city": is_city})
            offset += len(page["results"])
            if page["endOfRecords"] or not page["results"]:
                break
    gadm = pd.DataFrame(rows)

    counties = gpd.read_file(RAW / "cb_2020_us_county_20m.zip")
    census = pd.DataFrame({"fips": counties["GEOID"], "state_name": counties["STATE_NAME"],
                           "name": counties["NAME"], "is_city": counties["LSAD"] == "25"})
    for df in (gadm, census):
        df["key"] = df["name"].map(_norm_county_name)
    # Independent cities only need the city flag where a same-named county exists.
    m = gadm.merge(census[["fips", "state_name", "key", "is_city"]], on=["state_name", "key", "is_city"], how="left")
    unmatched = m["fips"].isna()
    if unmatched.any():
        retry = gadm[unmatched.values].merge(census[["fips", "state_name", "key"]], on=["state_name", "key"], how="left")
        retry = retry.drop_duplicates("gadm_gid", keep=False)  # only accept unambiguous matches
        m = m.set_index("gadm_gid")
        m.loc[retry["gadm_gid"], "fips"] = retry.set_index("gadm_gid")["fips"]
        m = m.reset_index()

    # Renamed or merged since GADM's boundaries; lakes and old Alaska areas stay unmatched.
    m.loc[m["gadm_gid"] == "USA.42.56_1", "fips"] = "46102"  # Shannon -> Oglala Lakota, SD
    m.loc[m["gadm_gid"] == "USA.47.29_1", "fips"] = "51005"  # Clifton Forge -> Alleghany, VA

    m.to_csv(path, index=False)
    return m


def _norm_county_name(name):
    n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    n = n.replace("saint ", "st. ").replace("ste. ", "sainte ")
    for drop in (" city", " county", " parish", " borough", " census area", ".", "'", "-", " "):
        n = n.replace(drop, "")
    return n
