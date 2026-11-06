"""Part 1: spatial distribution of activities.
Activities = destinations of trips inside the study area (work, education, shop, leisure, other).
Compared on a 5 km grid (all persons and by age group) and by distance from Fribourg city centre."""
import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import PowerNorm, Normalize
from matplotlib.patches import Rectangle

import plot_style as ps
from config import CLEAN_DIR, DATA, load_boundary
from compare_chains import AGE_BINS, AGE_LABELS

OUT = DATA / "results"
PURPOSES = ["work", "education", "shop", "leisure", "other"]
CENTRE = (2578500, 1183800)            # Fribourg city centre (LV95)
RING_EDGES = [0, 2, 5, 10, 20, 1000]   # km from the centre
RING_LABELS = ["0-2", "2-5", "5-10", "10-20", "20+"]
CELL = 5000                            # grid cell size in metres
MIN_PERSONS = 3                        # MZ cells with fewer respondents are not drawn
CANTON_FILE = next((DATA / "stadtland_typologie").rglob("g1k23.shp"))   # BFS cantons 2023
BASEMAP = ("https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-grau/default/current/3857/"
           "{z}/{x}/{y}.jpeg")   # swisstopo Landeskarte grau, credit: (c) swisstopo


def load(name):
    persons = pd.read_csv(CLEAN_DIR / name / "persons.csv")
    persons = persons[persons["in_mz_age_range"]]
    trips = pd.read_csv(CLEAN_DIR / name / "trips.csv")
    acts = trips.merge(persons[["person_id", "weight", "age"]], on="person_id")
    acts = acts[acts["destination_purpose"].isin(PURPOSES)].dropna(subset=["destination_x"]).copy()
    acts["age_group"] = pd.cut(acts["age"], AGE_BINS, right=False, labels=AGE_LABELS)
    dist_km = np.hypot(acts["destination_x"] - CENTRE[0], acts["destination_y"] - CENTRE[1]) / 1000
    acts["ring"] = pd.cut(dist_km, RING_EDGES, right=False, labels=RING_LABELS)
    acts["cell_x"] = (acts["destination_x"] // CELL).astype(int)
    acts["cell_y"] = (acts["destination_y"] // CELL).astype(int)
    return acts


def shares(acts, by):
    s = acts.groupby(by, observed=False)["weight"].sum()
    return 100 * s / s.sum()


def mismatch(a, b):
    both = pd.concat([a, b], axis=1).fillna(0)
    return 0.5 * (both.iloc[:, 0] - both.iloc[:, 1]).abs().sum()


def ring_table(ms, mz, groups):
    """Share of activities per distance ring (numbers for the text, no figure)."""
    rows = []
    for label, select in groups:
        p, m = shares(select(ms), "ring"), shares(select(mz), "ring")
        rows.append({"group": label, "mismatch %": round(mismatch(p, m), 1),
                     **{f"synthetic {r}": round(p[r], 1) for r in RING_LABELS},
                     **{f"MZ {r}": round(m[r], 1) for r in RING_LABELS}})
    return pd.DataFrame(rows)


def grid_setup(boundary):
    """Grid cells covering the study area, and the canton outline."""
    minx, miny, maxx, maxy = boundary.bounds
    xs = np.arange(int(minx // CELL), int(maxx // CELL) + 1)
    ys = np.arange(int(miny // CELL), int(maxy // CELL) + 1)
    edges_x = np.append(xs, xs[-1] + 1) * CELL
    edges_y = np.append(ys, ys[-1] + 1) * CELL
    canton = gpd.read_file(CANTON_FILE)
    canton = canton[canton["KTNR"] == 10]                    # 10 = canton of Fribourg
    return xs, ys, edges_x, edges_y, canton


def as_grid(values, xs, ys):
    """Series indexed by (cell_x, cell_y) -> 2D array for pcolormesh."""
    arr = np.full((len(ys), len(xs)), np.nan)
    for (cx, cy), v in values.items():
        if xs[0] <= cx <= xs[-1] and ys[0] <= cy <= ys[-1]:
            arr[cy - ys[0], cx - xs[0]] = v
    return arr


def draw_map(ax, values, hide, xs, ys, edges_x, edges_y, canton, boundary, cmap, norm, title):
    """One map panel: basemap, coloured cells, hatched hidden cells, borders and centre."""
    ax.set_xlim(edges_x[0], edges_x[-1])
    ax.set_ylim(edges_y[0], edges_y[-1])
    ax.set_aspect("equal")
    ctx.add_basemap(ax, crs="EPSG:2056", source=BASEMAP, zoom=10, attribution=False, zorder=0)
    mesh = ax.pcolormesh(edges_x, edges_y, as_grid(values, xs, ys), cmap=cmap, norm=norm,
                         alpha=0.75, zorder=1)
    if hide is not None:
        for cx, cy in hide.index[hide]:
            ax.add_patch(Rectangle((cx * CELL, cy * CELL), CELL, CELL, facecolor=ps.HIDDEN_FILL,
                                   edgecolor=ps.HIDDEN_EDGE, hatch="////", linewidth=0,
                                   alpha=0.8, zorder=2))
    canton.boundary.plot(ax=ax, color="black", linewidth=0.5, zorder=3)
    gpd.GeoSeries([boundary]).boundary.plot(ax=ax, color="#555555", linewidth=0.4,
                                            linestyle="--", zorder=3)
    ax.plot(*CENTRE, marker="*", color="black", markersize=5, zorder=4)
    ax.set_axis_off()
    ax.set_title(title)
    return mesh


def compare_cells(p_acts, m_acts):
    """Shares per cell, their difference, and which cells have too few MZ respondents."""
    p, m = shares(p_acts, ["cell_x", "cell_y"]), shares(m_acts, ["cell_x", "cell_y"])
    both = pd.concat([p.rename("synthetic"), m.rename("MZ")], axis=1).fillna(0)
    both["diff"] = both["synthetic"] - both["MZ"]
    respondents = m_acts.groupby(["cell_x", "cell_y"])["person_id"].nunique().reindex(both.index).fillna(0)
    hide = (respondents > 0) & (respondents < MIN_PERSONS)
    return both, hide, mismatch(p, m)


def grid_figure(ms, mz, boundary):
    """All persons: synthetic share, MZ share and difference per 5 km cell."""
    xs, ys, edges_x, edges_y, canton = grid_setup(boundary)
    both, hide, mm = compare_cells(ms, mz)
    vmax = both[["synthetic", "MZ"]].mask(hide, axis=0).max().max()
    share_norm = PowerNorm(gamma=0.5, vmin=0, vmax=vmax)     # square-root stretch: small shares stay visible
    diff_norm = Normalize(vmin=-5, vmax=5)                   # capped at +-5 pp, same as the age maps

    fig, axes = plt.subplots(1, 3, figsize=(ps.TEXT_WIDTH, 3.2))
    panels = [("Synthetic population", both["synthetic"], None, ps.SHARE_CMAP, share_norm,
               "share of activities [%]", "neither"),
              ("Mikrozensus (weighted)", both["MZ"].mask(hide), hide, ps.SHARE_CMAP, share_norm,
               "share of activities [%]", "neither"),
              ("Difference", both["diff"].mask(hide), hide, ps.DIFF_CMAP, diff_norm,
               "difference [percentage points]", "both")]
    for ax, (title, values, hidden, cmap, norm, unit, extend) in zip(axes, panels):
        mesh = draw_map(ax, values, hidden, xs, ys, edges_x, edges_y, canton, boundary, cmap, norm, title)
        fig.colorbar(mesh, ax=ax, orientation="horizontal", fraction=0.05, pad=0.03,
                     extend=extend, label=unit)
    fig.tight_layout()
    ps.save(fig, OUT, "activities_grid")
    return mm, int(hide.sum())


def age_grid_figure(ms, mz, boundary):
    """Difference maps (synthetic - MZ) per age group on the same grid and colour scale."""
    xs, ys, edges_x, edges_y, canton = grid_setup(boundary)
    norm = Normalize(vmin=-5, vmax=5)   # capped at +-5 pp so smaller differences stay visible

    fig, axes = plt.subplots(2, 2, figsize=(ps.TEXT_WIDTH, 8.0))
    for i, (ax, g) in enumerate(zip(axes.flat, AGE_LABELS)):
        m_g = mz[mz["age_group"] == g]
        both, hide, _ = compare_cells(ms[ms["age_group"] == g], m_g)
        print(f"Age {g}: {int(hide.sum())} of {len(both)} cells hidden "
              f"({m_g['person_id'].nunique()} respondents)")
        ax.set_anchor("E" if i % 2 == 0 else "W")   # left column to the right, right column to the left
        mesh = draw_map(ax, both["diff"].mask(hide), hide, xs, ys, edges_x, edges_y, canton, boundary,
                        ps.DIFF_CMAP, norm, f"Age {g}")

    fig.colorbar(mesh, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.04, pad=0.04,
                 extend="both", label="difference synthetic population - Mikrozensus [percentage points]")
    ps.save(fig, OUT, "activities_grid_by_age")


def main():
    ps.apply()
    OUT.mkdir(parents=True, exist_ok=True)
    boundary = load_boundary()
    ms, mz = load("matsim"), load("microcensus")
    print(f"Activities compared: synthetic {len(ms)} | MZ {len(mz)} "
          f"(from {mz['person_id'].nunique()} respondents)")

    grid_mm, n_hidden = grid_figure(ms, mz, boundary)
    print(f"Grid mismatch (5 km cells, all activities): {grid_mm:.1f}% ({n_hidden} MZ cells hidden)")
    age_grid_figure(ms, mz, boundary)

    purpose_groups = [("all activities", lambda a: a)] + [
        (p, lambda a, p=p: a[a["destination_purpose"] == p]) for p in PURPOSES]
    age_groups = [(f"age {g}", lambda a, g=g: a[a["age_group"] == g]) for g in AGE_LABELS]

    pd.set_option("display.width", 250)
    print("\nShare of activities by distance from the centre (%), for the text:")
    print(ring_table(ms, mz, purpose_groups + age_groups).to_string(index=False))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()