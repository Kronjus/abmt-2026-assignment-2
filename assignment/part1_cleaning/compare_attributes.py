"""Part 1: socio-demographic and spatial attributes, MATSim vs Mikrozensus (persons aged 6+)."""
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plot_style as ps

from config import CLEAN_DIR, DATA, CRS_LV95
from categories import AGE_LABELS, CAR_CLASSES, HH_TYPES, SUBSCRIPTIONS

OUT = DATA / "results"
SETTLEMENT_FILE = DATA / "stadtland_typologie" / "settlement_2023.gpkg"
PT_FILE = DATA / "gueteklassen_oev_2026" / "OeV_Gueteklassen_ARE.gpkg"
SETTLEMENTS = ["urban", "intermediate", "rural"]
PT_CLASSES = ["A", "B", "C", "D", "none"]


def homes(p):
    return gpd.GeoDataFrame(index=p.index, crs=CRS_LV95,
                            geometry=gpd.points_from_xy(p["home_x"], p["home_y"]))


def add_settlement(p):
    """Stadt/Land class of the home municipality. Homes in unclassified areas (lakes) -> nearest."""
    zones = gpd.read_file(SETTLEMENT_FILE).dropna(subset=["settlement"])[["settlement", "geometry"]]
    pts = homes(p)
    found = pts.sjoin(zones, how="left", predicate="within")["settlement"].groupby(level=0).first()
    missing = found.index[found.isna()]
    if len(missing):
        near = pts.loc[missing].sjoin_nearest(zones, how="left")
        found[missing] = near["settlement"].groupby(level=0).first()
    p["settlement"] = found
    return len(missing)


def add_pt_quality_are(p):
    """ARE PT quality class at the home location; outside every polygon -> 'none'.
    Overlapping polygons: keep the best class (A < B < C < D)."""
    zones = gpd.read_file(PT_FILE, layer="OeV_Gueteklassen_ARE", columns=["KLASSE"]).to_crs(CRS_LV95)
    joined = homes(p).sjoin(zones, how="left", predicate="within")
    p["pt_quality"] = joined["KLASSE"].groupby(level=0).min().reindex(p.index).fillna("none")


def shares(df, col, categories):
    totals = df.groupby(col)["weight"].sum().reindex(categories).fillna(0)
    return 100 * totals / totals.sum()


ATTRIBUTES = [  # (title, column, categories, kind)
    ("Age class", "age_class", AGE_LABELS[1:], "socio-demographic"),
    ("Car availability", "car_avail", CAR_CLASSES, "socio-demographic"),
    ("PT pass", "subscription", SUBSCRIPTIONS, "socio-demographic"),
    ("PT quality at home", "pt_quality", PT_CLASSES, "spatial"),
    ("Settlement type at home", "settlement", SETTLEMENTS, "spatial"),
]


def main():
    ps.apply()
    ms = pd.read_csv(CLEAN_DIR / "matsim" / "persons.csv")
    ms = ms[ms["in_mz_age_range"]].copy()
    mz = pd.read_csv(CLEAN_DIR / "microcensus" / "persons.csv")

    for name, df in [("MATSim", ms), ("MZ", mz)]:
        print(f"{name}: {add_settlement(df)} homes outside a classified municipality -> nearest one")

    if PT_FILE.exists():
        add_pt_quality_are(ms)
        add_pt_quality_are(mz)   # both datasets classified with the ARE map of 2026
        print("PT quality: both datasets classified with the ARE map of 2026")
    else:
        print("WARNING: ARE file not found - MATSim PT quality uses eqasim's own 'ovgk' attribute for now")

    fig, axes = plt.subplots(2, 3, figsize=(ps.TEXT_WIDTH, 5.0))
    rows = []
    for ax, (title, col, cats, kind) in zip(axes.flat, ATTRIBUTES):
        p, m = shares(ms, col, cats), shares(mz, col, cats)
        mm = 0.5 * (p - m).abs().sum()
        n_missing = int(mz[col].isna().sum())
        for c in cats:
            rows.append({"attribute": title, "type": kind, "category": c, "MATSim %": round(p[c], 1),
                         "MZ % (weighted)": round(m[c], 1), "diff pp": round(p[c] - m[c], 1)})
        rows.append({"attribute": title, "type": kind, "category": "MISMATCH (pp)", "diff pp": round(mm, 1)})

        x = np.arange(len(cats))
        ax.bar(x - 0.2, p.values, 0.4, color=ps.MATSIM, label=ps.LABELS["matsim"])
        ax.bar(x + 0.2, m.values, 0.4, color=ps.MZ, label=ps.LABELS["mz"])
        ax.set_xticks(x, cats, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_ylabel("share of persons [%]")
        ps.bar_grid(ax)
        ax.set_title(f"{title}\nmismatch {mm:.1f}%")
        if n_missing:
            print(f"{title}: {n_missing} MZ persons with missing value (for the caption)")
    for ax in axes.flat[len(ATTRIBUTES):]:
        ax.set_visible(False)
    ps.shared_legend(fig, axes.flat[0])
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    ps.save(fig, OUT, "attributes")
    print(f"Sample sizes (for the caption): MATSim n = {len(ms)}, Mikrozensus n = {len(mz)}")

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "attributes.csv", index=False)
    pd.set_option("display.width", 200)
    print()
    print(table.fillna("").to_string(index=False))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()