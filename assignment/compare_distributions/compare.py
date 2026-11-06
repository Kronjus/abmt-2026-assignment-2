"""Compare the filtered synthetic population with the filtered microcensus.

Three socio-demographic attributes (age, car availability, PT subscription) and
two spatial ones (PT quality at home, settlement pattern), as shares of persons.

Usage (from the repository root, after assignment/filter_data/filter_data.py):
    .venv/Scripts/python assignment/compare_distributions/compare.py

Outputs: assignment/compare_distributions/output/comparison.csv, comparison.png and mismatch.csv

Each subplot title shows the mismatch: the total variation distance between the two distributions,
0.5 * sum(|population% - microcensus%|), in percentage points (0 = identical, 100 = disjoint). It is the
share of persons that would have to change category for the population to match the microcensus. Next to it
is the sampling noise of the microcensus: the mean and 95th percentile of the TVD of bootstrap resamples of
the microcensus persons to the full sample (bootstrap_noise.py). A mismatch below the 95th percentile cannot
be told apart from sampling noise.

Notes
- Microcensus shares are weighted with the person weight WP; population shares are unweighted.
- Car availability: the population only distinguishes always/never, so the microcensus "always" and
  "by arrangement" answers are merged into "available".
- Persons aged under 6 are dropped from the population: the microcensus has no target persons that young.
- PT quality: both datasets are looked up from the same ARE Gueteklassen polygons (EPSG:2056) at the home
  location; homes outside every polygon get "none". Microcensus homes (W_X/W_Y) are WGS84 and are transformed.
  The microcensus' own W_OeV_KLASSE is not used.
- Settlement pattern: both datasets are classified from the home location with the BFS Stadt/Land-
  Typologie 2012 (STALAN2012: urban / intermediate / rural municipality). Population homes are in
  EPSG:2056, microcensus homes (W_X/W_Y) are WGS84 and are transformed.
- PT subscription: the population has one attribute (GA, HT, VA, VA+HT, none). The microcensus asks for each
  subscription separately (zielpersonen.csv, 1 = yes): GA = f41610a; HT = Halbtax f41610b; VA = Verbund-/Interabo
  f41610c or Streckenabo f41610d. GA wins over everything else, HT + VA gives "VA+HT". Juniorkarte, Gleis 7 and
  other subscriptions are not modelled and count as none, as does "not asked" (-99).
"""
import gzip
import re
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bootstrap_noise import tvd_noise

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
POPULATION = DATA / "filtered" / "fribourg_population.xml.gz"
MC = DATA / "filtered" / "microcensus"
GUETEKLASSEN = DATA / "gueteklassen_oev_2026" / "OeV_Gueteklassen_ARE.gpkg"
STADTLAND = DATA / "stadtland_typologie" / "Boundaries_G1_GDETYP2012_20230101_de.gpkg"
OUT = Path(__file__).resolve().parent / "output"

AGE_BINS = [6, 18, 30, 45, 65, 80, 200]
AGE_LABELS = ["6-17", "18-29", "30-44", "45-64", "65-79", "80+"]
CAR = ["available", "never"]
SUB = ["GA", "HT", "VA", "VA+HT", "none"]
PT = ["A", "B", "C", "D", "none"]
SETTLEMENT = ["urban", "intermediate", "rural"]
STALAN = {"1": "urban", "2": "intermediate", "3": "rural"}  # stored as text in the gpkg
ATTRIBUTES = [  # (title, categories, kind)
    ("Age", AGE_LABELS, "socio-demographic"),
    ("Car availability", CAR, "socio-demographic"),
    ("PT subscription", SUB, "socio-demographic"),
    ("PT quality at home", PT, "spatial"),
    ("Settlement pattern", SETTLEMENT, "spatial"),
]


def read_population():
    attrs = {"age": "age", "carAvail": "car", "subscriptions": "sub", "home_coordinate_x": "x", "home_coordinate_y": "y"}
    rows = []
    block = None
    with gzip.open(POPULATION, "rt", encoding="utf-8") as f:
        for line in f:
            if block is None:
                m = re.search(r'<person id="([^"]+)"', line)
                if m:
                    block, row = [], {"id": m.group(1)}
            else:
                if "<plan" in line:  # person attributes end where the plans start
                    rows.append(row)
                    block = None
                    continue
                m = re.search(r'name="([^"]+)"[^>]*>([^<]*)<', line)
                if m and m.group(1) in attrs:
                    row[attrs[m.group(1)]] = m.group(2)
    pop = pd.DataFrame(rows).set_index("id")
    pop["age"] = pop["age"].astype(int)
    pop[["x", "y"]] = pop[["x", "y"]].astype(float)
    return pop


def pt_quality(x, y, crs):
    """ARE OeV-Gueteklasse of each point, "none" outside every polygon (index of x/y is kept)."""
    zones = gpd.read_file(GUETEKLASSEN, layer="OeV_Gueteklassen_ARE", columns=["KLASSE"]).to_crs("EPSG:2056")
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y), index=x.index, crs=crs).to_crs("EPSG:2056")
    joined = pts.sjoin(zones, how="left", predicate="within")
    best = joined["KLASSE"].groupby(level=0).min()  # overlapping polygons: keep the best class (A < B < ...)
    return best.reindex(x.index).fillna("none")


def settlement_type(x, y, crs):
    """BFS Stadt/Land-Typologie class of each point (index of x/y is kept)."""
    zones = gpd.read_file(STADTLAND, columns=["STALAN2012"]).to_crs("EPSG:2056")
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y), index=x.index, crs=crs).to_crs("EPSG:2056")
    joined = pts.sjoin(zones, how="left", predicate="within")
    found = joined["STALAN2012"].groupby(level=0).first().reindex(x.index)
    missing = found.index[found.isna()]  # e.g. points on an unclassified polygon: take the nearest classified one
    if len(missing):
        near = pts.loc[missing].sjoin_nearest(zones.dropna(subset=["STALAN2012"]), how="left")
        found[missing] = near["STALAN2012"].groupby(level=0).first()
        print(f"settlement: {len(missing)} points outside a classified municipality assigned to the nearest one")
    return found.map(STALAN)


def share(series, categories, weights=None):
    w = weights if weights is not None else pd.Series(1.0, index=series.index)
    total = w.groupby(series).sum().reindex(categories).fillna(0.0)
    return 100 * total / total.sum()


def read_microcensus():
    z = pd.read_csv(MC / "zielpersonen.csv", encoding="latin1", usecols=["HHNR", "alter", "f42100e", "WP", "f41610a", "f41610b", "f41610c", "f41610d"])
    h = pd.read_csv(MC / "haushalte.csv", encoding="latin1", usecols=["HHNR", "W_OeV_KLASSE", "W_X", "W_Y"])
    mc = z.merge(h, on="HHNR")
    mc["age_c"] = pd.cut(mc["alter"], AGE_BINS, right=False, labels=AGE_LABELS)
    mc["car"] = mc["f42100e"].map({1: "available", 2: "available"}).fillna("never")  # 3 and -99 (n/a) = never
    ga, ht = mc["f41610a"] == 1, mc["f41610b"] == 1
    va = (mc["f41610c"] == 1) | (mc["f41610d"] == 1)
    mc["sub"] = np.select([ga, ht & va, ht, va], ["GA", "VA+HT", "HT", "VA"], "none")
    mc["pt"] = pt_quality(mc["W_X"], mc["W_Y"], "EPSG:4326")
    mc["settlement"] = settlement_type(mc["W_X"], mc["W_Y"], "EPSG:4326")
    return mc


def main():
    pop = read_population()
    pop["pt"] = pt_quality(pop["x"], pop["y"], "EPSG:2056")
    pop["settlement"] = settlement_type(pop["x"], pop["y"], "EPSG:2056")
    pop = pop[pop["age"] >= 6].copy()
    pop["age_c"] = pd.cut(pop["age"], AGE_BINS, right=False, labels=AGE_LABELS)
    pop["car"] = pop["car"].map({"always": "available", "arrangement": "available", "never": "never"})
    mc = read_microcensus()
    print(f"population: {len(pop)} persons; microcensus: {len(mc)} target persons")

    columns = {"Age": "age_c", "Car availability": "car", "PT subscription": "sub",
               "PT quality at home": "pt", "Settlement pattern": "settlement"}
    table, summary = [], []
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    for ax, (title, cats, kind) in zip(axes, ATTRIBUTES):
        col = columns[title]
        assert pop[col].notna().all(), f"unmapped population values in {title}"
        if mc[col].isna().any():  # e.g. unknown codes
            print(f"microcensus: {mc[col].isna().sum()} persons without {title}, excluded from its shares")
        p, m = share(pop[col], cats), share(mc[col], cats, mc["WP"])
        tvd = 0.5 * (p - m).abs().sum()  # total variation distance: share of persons that would have to move
        counts = pd.get_dummies(mc[col]).reindex(columns=cats, fill_value=False).astype(float).mul(mc["WP"], axis=0)
        noise_mean, noise_p95 = tvd_noise(counts)
        summary.append({"attribute": title, "type": kind, "total_variation_distance_pp": tvd,
                        "noise_mean_pp": noise_mean, "noise_p95_pp": noise_p95})
        for c in cats:
            table.append({"type": kind, "attribute": title, "category": c, "population_%": p[c],
                          "microcensus_%": m[c], "difference_pp": p[c] - m[c]})
        x = range(len(cats))
        ax.bar([i - 0.2 for i in x], p.values, 0.4, label="population")
        ax.bar([i + 0.2 for i in x], m.values, 0.4, label="microcensus (weighted)")
        ax.set_xticks(list(x), cats, rotation=30, ha="right")
        ax.set_title(f"{title} ({kind})\nmismatch: {tvd:.1f} pp (noise: {noise_mean:.1f} mean, {noise_p95:.1f} p95)")
        ax.set_ylabel("share of persons [%]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))  # leave room for the legend below the plots
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "comparison.png", dpi=150)
    df = pd.DataFrame(table)
    df.to_csv(OUT / "comparison.csv", index=False, float_format="%.2f")
    print(df.round(1).to_string(index=False))
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "mismatch.csv", index=False, float_format="%.2f")
    print(summary.round(1).to_string(index=False))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
