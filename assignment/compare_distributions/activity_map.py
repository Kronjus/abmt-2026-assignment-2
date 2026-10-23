"""Compare the spatial distribution of activities of the filtered synthetic population with the filtered
microcensus, aggregated over all persons and by age group.

Usage (from the repository root, after assignment/filter_data/filter_data.py):
    .venv/Scripts/python assignment/compare_distributions/activity_map.py

Outputs in assignment/compare_distributions/output/: activity_map.png (all activities),
activity_map_no_home.png (home activities excluded), activity_map.csv and activity_map_mismatch.csv

Notes
- An activity is the destination of a trip. Population: every activity of the plan except the first one
  (the start-of-day home, which has no trip leading to it). Microcensus: the destination (Z_X/Z_Y, WGS84)
  of every stage that ends a Weg, with the purpose from f52900. Activity types and purposes are mapped as
  in activity_chains.py. `outside` activities (population) and unknown purposes (microcensus) are skipped.
- Activities are counted in 1 km cells (EPSG:2056) and only those inside the study-area boundary are kept.
  The map shows each cell's share of all activities of that dataset and group, so datasets of different
  size are comparable. Microcensus shares are weighted with the person weight WP.
- Persons aged under 6 are dropped, as in activity_chains.py.
- The basemap tiles (swisstopo Landeskarte grau, free geo.admin.ch WMTS) are downloaded when the script runs.
- The mismatch is the total variation distance between the two cell distributions, in percentage points.
"""
import gzip
import re
from pathlib import Path

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from pyproj import Transformer
from shapely import points

from activity_chains import AGE_BINS, AGE_LABELS, MC_PURPOSE, MC, OUT, POP_TYPE, POPULATION, ROOT

BOUNDARY = ROOT / "data" / "Boundary" / "fribourg_convexhull.shp"
CELL = 1000  # m
BASEMAP = "https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-grau/default/current/3857/{z}/{x}/{y}.jpeg"  # grey, so the cell colours stay readable


def read_population():
    rows, person, age, first = [], None, None, True
    outside = 0
    with gzip.open(POPULATION, "rt", encoding="utf-8") as f:
        for line in f:
            if "<person " in line:
                person, age, first = True, None, True
                continue
            if person is None:
                continue
            m = re.search(r'name="age"[^>]*>(\d+)<', line)
            if m:
                age = int(m.group(1))
            m = re.search(r'<activity type="([^"]+)".* x="([^"]+)" y="([^"]+)"', line)
            if m and not m.group(1).endswith(" interaction"):
                if first:  # start-of-day activity, no trip leads to it
                    first = False
                elif m.group(1) == "outside":
                    outside += 1
                else:
                    rows.append({"age": age, "type": POP_TYPE[m.group(1)],
                                 "x": float(m.group(2)), "y": float(m.group(3))})
    print(f"population: {outside} outside activities skipped")
    return pd.DataFrame(rows)


def read_microcensus():
    z = pd.read_csv(MC / "zielpersonen.csv", encoding="latin1", usecols=["HHNR", "alter", "WP"])
    e = pd.read_csv(MC / "etappen.csv", encoding="latin1", usecols=["HHNR", "WEGNR", "ETNR", "f52900", "Z_X", "Z_Y"])
    last = e.sort_values("ETNR").groupby(["HHNR", "WEGNR"]).tail(1)
    unknown = ~last["f52900"].isin(MC_PURPOSE)
    print(f"microcensus: {unknown.sum()} trips with unknown purpose skipped")
    last = last[~unknown].copy()
    last["x"], last["y"] = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True).transform(
        last["Z_X"].to_numpy(), last["Z_Y"].to_numpy())
    last["type"] = last["f52900"].map(MC_PURPOSE)
    df = last.merge(z, on="HHNR")
    return df.rename(columns={"alter": "age", "WP": "weight"})[["age", "type", "x", "y", "weight"]]


def grid_share(df, x_edges, y_edges, weights=None):
    w = weights if weights is not None else np.ones(len(df))
    h, _, _ = np.histogram2d(df["x"], df["y"], bins=[x_edges, y_edges], weights=w)
    return 100 * h.T / h.sum()  # rows = y, columns = x


def draw(shares, boundary, extent, title, path):
    groups = list(shares)
    fig, axes = plt.subplots(len(groups), 3, figsize=(15, 4.2 * len(groups)), squeeze=False)
    for row, group in zip(axes, groups):
        p, m, n = shares[group]
        diff = p - m
        vmax = max(np.nanpercentile(np.r_[p.ravel(), m.ravel()], 99.5), 1e-9)
        dmax = max(np.nanpercentile(np.abs(diff[(p > 0) | (m > 0)]), 99.5), 1e-9)
        panels = [(p, "population", "viridis", None, "share of activities [%]"),
                  (m, "microcensus (weighted)", "viridis", None, "share of activities [%]"),
                  (diff, "population - microcensus", "RdBu_r", TwoSlopeNorm(0, -dmax, dmax), "difference [pp]")]
        for ax, (data, name, cmap, norm, label) in zip(row, panels):
            data = np.ma.masked_where((p == 0) & (m == 0), data)
            kw = {"norm": norm} if norm else {"vmin": 0, "vmax": vmax}
            # opacity grows with the magnitude, so near-zero cells let the basemap show through
            alpha = 0.1 + 0.9 * np.clip(np.abs(data.filled(0)) / (dmax if norm else vmax), 0, 1)
            img = ax.imshow(data, origin="lower", extent=extent, cmap=cmap, alpha=alpha, zorder=2, **kw)
            cx.add_basemap(ax, crs="EPSG:2056", source=BASEMAP, zoom=12, zorder=1, attribution=False)
            boundary.boundary.plot(ax=ax, color="black", linewidth=0.8, zorder=3)
            ax.set_title(f"{group}: {name}\n(n = {n[0] if name == 'population' else n[1]})"
                         if name != "population - microcensus" else f"{group}: {name}")
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(img, ax=ax, shrink=0.8, label=label)
    fig.suptitle(title, fontsize=14)
    fig.text(0.99, 0.003, "Basemap: © swisstopo (Landeskarte grau)", ha="right", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run(pop, mc, boundary, x_edges, y_edges, name, title, summary, table):
    extent = (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1])
    groups = [("All persons", pop, mc)] + [
        (f"Age {a}", pop[pop["age_c"] == a], mc[mc["age_c"] == a]) for a in AGE_LABELS]
    shares = {}
    for title_g, p_df, m_df in groups:
        p = grid_share(p_df, x_edges, y_edges)
        m = grid_share(m_df, x_edges, y_edges, m_df["weight"])
        shares[title_g] = (p, m, (len(p_df), len(m_df)))
        summary.append({"figure": name, "group": title_g, "activities_population": len(p_df),
                        "activities_microcensus": len(m_df), "total_variation_distance_pp": 0.5 * np.abs(p - m).sum()})
        for iy, ix in zip(*np.nonzero((p > 0) | (m > 0))):
            table.append({"figure": name, "group": title_g, "x_min": x_edges[ix], "y_min": y_edges[iy],
                          "population_%": p[iy, ix], "microcensus_%": m[iy, ix], "difference_pp": p[iy, ix] - m[iy, ix]})
    draw(shares, boundary, extent, title, OUT / f"{name}.png")


def main():
    pop, mc = read_population(), read_microcensus()
    boundary = gpd.read_file(BOUNDARY).set_crs("EPSG:2056", allow_override=True)
    hull = boundary.geometry.union_all()
    minx, miny, maxx, maxy = hull.bounds
    x_edges = np.arange(np.floor(minx / CELL) * CELL, maxx + CELL, CELL)
    y_edges = np.arange(np.floor(miny / CELL) * CELL, maxy + CELL, CELL)

    pop, mc = pop[pop["age"] >= 6].copy(), mc[mc["age"] >= 6].copy()
    kept = []
    for label, df in (("population", pop), ("microcensus", mc)):
        inside = pd.Series(hull.covers(points(df["x"].to_numpy(), df["y"].to_numpy())), index=df.index)
        print(f"{label}: {int((~inside).sum())} of {len(df)} activities outside the boundary dropped")
        df = df[inside].copy()
        df["age_c"] = pd.cut(df["age"], AGE_BINS, right=False, labels=AGE_LABELS)
        kept.append(df)
    pop, mc = kept

    summary, table = [], []
    run(pop, mc, boundary, x_edges, y_edges, "activity_map",
        f"Spatial distribution of activities ({CELL // 1000} km cells), all activity types", summary, table)
    run(pop[pop["type"] != "H"], mc[mc["type"] != "H"], boundary, x_edges, y_edges, "activity_map_no_home",
        f"Spatial distribution of activities ({CELL // 1000} km cells), home activities excluded", summary, table)

    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(table)
    table.to_csv(OUT / "activity_map.csv", index=False, float_format="%.4f")
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "activity_map_mismatch.csv", index=False, float_format="%.2f")
    print(summary.round(1).to_string(index=False))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
