"""Compare the filtered synthetic population with the filtered microcensus.

Three socio-demographic attributes (age, car availability, household type) and
two spatial ones (PT quality at home, settlement pattern), as shares of persons.

Usage (from the repository root, after assignment/filter_data/filter_data.py):
    .venv/Scripts/python assignment/compare_distributions/compare.py

Outputs: assignment/compare_distributions/output/comparison.csv, comparison.png and mismatch.csv

Each subplot title shows the mismatch: the total variation distance between the two distributions,
0.5 * sum(|population% - microcensus%|), in percentage points (0 = identical, 100 = disjoint). It is the
share of persons that would have to change category for the population to match the microcensus.

Notes
- Microcensus shares are weighted with the person weight WP; population shares are unweighted.
- Car availability: the population only distinguishes always/never, so the microcensus "always" and
  "by arrangement" answers are merged into "available".
- The microcensus has no target persons under 6, so the youngest age class differs by design.
- PT quality: the population value is looked up from the ARE Gueteklassen polygons (EPSG:2056) at
  the home location; homes outside every polygon get "none". The microcensus uses its own W_OeV_KLASSE.
- Settlement pattern: both datasets are classified from the home location with the BFS Stadt/Land-
  Typologie 2012 (STALAN2012: urban / intermediate / rural municipality). Population homes are in
  EPSG:2056, microcensus homes (W_X/W_Y) are WGS84 and are transformed.
- Household type: the population has no such attribute, so it is derived from the household
  members (size, and whether a member is under 18).
"""
import gzip
import re
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
POPULATION = DATA / "filtered" / "fribourg_population.xml.gz"
HOUSEHOLDS = DATA / "Fribourg_10pct" / "fribourg_households.xml.gz"
MC = DATA / "filtered" / "microcensus"
GUETEKLASSEN = DATA / "gueteklassen_oev_2026" / "OeV_Gueteklassen_ARE.gpkg"
STADTLAND = DATA / "stadtland_typologie" / "Boundaries_G1_GDETYP2012_20230101_de.gpkg"
OUT = Path(__file__).resolve().parent / "output"

AGE_BINS = [0, 18, 30, 45, 65, 80, 200]
AGE_LABELS = ["0-17", "18-29", "30-44", "45-64", "65-79", "80+"]
CAR = ["available", "never"]
HH_TYPE = ["single person", "couple, no children", "with children", "other"]
PT = ["A", "B", "C", "D", "none"]
SETTLEMENT = ["urban", "intermediate", "rural"]
STALAN = {"1": "urban", "2": "intermediate", "3": "rural"}  # stored as text in the gpkg
ATTRIBUTES = [  # (title, categories, kind)
    ("Age", AGE_LABELS, "socio-demographic"),
    ("Car availability", CAR, "socio-demographic"),
    ("Household type", HH_TYPE, "socio-demographic"),
    ("PT quality at home", PT, "spatial"),
    ("Settlement pattern", SETTLEMENT, "spatial"),
]


def read_population():
    attrs = {"age": "age", "carAvail": "car", "home_x": "x", "home_y": "y"}
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


def add_pt_quality(pop):
    zones = gpd.read_file(GUETEKLASSEN, layer="OeV_Gueteklassen_ARE", columns=["KLASSE"])
    homes = gpd.GeoDataFrame(geometry=gpd.points_from_xy(pop["x"], pop["y"]), index=pop.index, crs="EPSG:2056")
    joined = homes.sjoin(zones.to_crs(homes.crs), how="left", predicate="within")
    best = joined["KLASSE"].groupby(level=0).min()  # overlapping polygons: keep the best class (A < B < ...)
    pop["pt"] = best.reindex(pop.index).fillna("none")


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


def add_household_type(pop):
    text = gzip.open(HOUSEHOLDS, "rt", encoding="utf-8").read()
    member_of = {}
    for hh in re.findall(r"<household .*?</household>", text, re.S):
        hid = re.search(r'<household id="([^"]+)"', hh).group(1)
        for pid in re.findall(r'refId="([^"]+)"', hh):
            member_of[pid] = hid
    pop["hh"] = pop.index.map(member_of)
    grouped = pop.groupby("hh")["age"]
    size = grouped.transform("size")  # members present in the filtered population
    has_child = grouped.transform(lambda a: (a < 18).any())
    kind = pd.Series("other", index=pop.index)
    kind[(size == 2) & ~has_child] = "couple, no children"
    kind[has_child] = "with children"
    kind[size == 1] = "single person"
    pop["hh_type"] = kind
    print(f"population: {pop['hh'].isna().sum()} persons without a household")


def share(series, categories, weights=None):
    w = weights if weights is not None else pd.Series(1.0, index=series.index)
    total = w.groupby(series).sum().reindex(categories).fillna(0.0)
    return 100 * total / total.sum()


def read_microcensus():
    z = pd.read_csv(MC / "zielpersonen.csv", encoding="latin1", usecols=["HHNR", "alter", "f42100e", "WP"])
    h = pd.read_csv(MC / "haushalte.csv", encoding="latin1", usecols=["HHNR", "hhtyp", "W_OeV_KLASSE", "W_X", "W_Y"])
    mc = z.merge(h, on="HHNR")
    mc["age_c"] = pd.cut(mc["alter"], AGE_BINS, right=False, labels=AGE_LABELS)
    mc["car"] = mc["f42100e"].map({1: "available", 2: "available"}).fillna("never")  # 3 and -99 (n/a) = never
    mc["hh_type"] = mc["hhtyp"].map({10: "single person", 210: "couple, no children", 220: "with children",
                                     230: "with children", 30: "other"})
    mc["pt"] = mc["W_OeV_KLASSE"].map({1: "A", 2: "B", 3: "C", 4: "D", 5: "none"})
    mc["settlement"] = settlement_type(mc["W_X"], mc["W_Y"], "EPSG:4326")
    return mc


def main():
    pop = read_population()
    add_household_type(pop)
    add_pt_quality(pop)
    pop["settlement"] = settlement_type(pop["x"], pop["y"], "EPSG:2056")
    pop["age_c"] = pd.cut(pop["age"], AGE_BINS, right=False, labels=AGE_LABELS)
    pop["car"] = pop["car"].map({"always": "available", "arrangement": "available", "never": "never"})
    mc = read_microcensus()
    print(f"population: {len(pop)} persons; microcensus: {len(mc)} target persons")

    columns = {"Age": "age_c", "Car availability": "car", "Household type": "hh_type",
               "PT quality at home": "pt", "Settlement pattern": "settlement"}
    table, summary = [], []
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    for ax, (title, cats, kind) in zip(axes, ATTRIBUTES):
        col = columns[title]
        assert pop[col].notna().all(), f"unmapped population values in {title}"
        if mc[col].isna().any():  # e.g. refused/unknown household type codes (-97, -98)
            print(f"microcensus: {mc[col].isna().sum()} persons without {title}, excluded from its shares")
        p, m = share(pop[col], cats), share(mc[col], cats, mc["WP"])
        tvd = 0.5 * (p - m).abs().sum()  # total variation distance: share of persons that would have to move
        summary.append({"attribute": title, "type": kind, "total_variation_distance_pp": tvd})
        for c in cats:
            table.append({"type": kind, "attribute": title, "category": c, "population_%": p[c],
                          "microcensus_%": m[c], "difference_pp": p[c] - m[c]})
        x = range(len(cats))
        ax.bar([i - 0.2 for i in x], p.values, 0.4, label="population")
        ax.bar([i + 0.2 for i in x], m.values, 0.4, label="microcensus (weighted)")
        ax.set_xticks(list(x), cats, rotation=30, ha="right")
        ax.set_title(f"{title} ({kind})\nmismatch: {tvd:.1f} pp")
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
