"""Part 1: spatial distribution of activities.
Activities = destinations of trips inside the study area (work, education, shop, leisure, other).
Compared on a 5 km grid and by distance from Fribourg city centre, aggregated and by age group."""
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import plot_style as ps
from matplotlib.colors import PowerNorm, Normalize

from config import CLEAN_DIR, DATA, load_boundary
from compare_chains import AGE_BINS, AGE_LABELS

OUT = DATA / "results"
PURPOSES = ["work", "education", "shop", "leisure", "other"]
CENTRE = (2578500, 1183800)            # Fribourg city centre (LV95)
RING_EDGES = [0, 2, 5, 10, 20, 1000]   # km from the centre
RING_LABELS = ["0-2", "2-5", "5-10", "10-20", "20+"]
CELL = 5000                            # grid cell size in metres
MIN_PERSONS = 5         # MZ cells with fewer respondents are not drawn
SETTLEMENT_FILE = DATA / "stadtland_typologie" / "settlement_2023.gpkg"   # for municipality lines
CANTON_FILE = next((DATA / "stadtland_typologie").rglob("g1k23.shp"))   # BFS cantons 2023


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


def ring_figure(ms, mz, groups, title, filename):
    """One panel per group: share of activities in each distance ring."""
    ncols = 3 if len(groups) > 4 else 2
    nrows = -(-len(groups) // ncols)          # rounds up
    fig, axes = plt.subplots(nrows, ncols, figsize=(ps.TEXT_WIDTH, 2.3 * nrows), sharey=True)
    rows = []
    for ax, (label, select) in zip(axes.flat, groups):
        p, m = shares(select(ms), "ring"), shares(select(mz), "ring")
        mm = mismatch(p, m)
        rows.append({"group": label, "mismatch pp": round(mm, 1),
                     **{f"MATSim {r}": round(p[r], 1) for r in RING_LABELS},
                     **{f"MZ {r}": round(m[r], 1) for r in RING_LABELS}})
        x = np.arange(len(RING_LABELS))
        ax.bar(x - 0.2, p.values, 0.4, color=ps.MATSIM, label=ps.LABELS["matsim"])
        ax.bar(x + 0.2, m.values, 0.4, color=ps.MZ, label=ps.LABELS["mz"])
        ax.set_xticks(x, RING_LABELS)
        ps.bar_grid(ax)
        ax.set_title(f"{title(label)} · {mm:.1f} pp")
    for ax in axes.flat[len(groups):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("share of activities [%]")
    for ax in axes[-1, :]:
        ax.set_xlabel("distance from centre [km]")
    ps.shared_legend(fig, axes.flat[0])
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    ps.save(fig, OUT, filename)
    return pd.DataFrame(rows)


def grid_figure(ms, mz, boundary):
    """Map: share of all activities per 5 km cell, MATSim vs MZ, and the difference."""
    p, m = shares(ms, ["cell_x", "cell_y"]), shares(mz, ["cell_x", "cell_y"])
    both = pd.concat([p.rename("MATSim"), m.rename("MZ")], axis=1).fillna(0)
    both["diff"] = both["MATSim"] - both["MZ"]

    # cells with only a few MZ respondents are hidden (privacy + noise)
    respondents = mz.groupby(["cell_x", "cell_y"])["person_id"].nunique().reindex(both.index).fillna(0)
    hide = (respondents > 0) & (respondents < MIN_PERSONS)
    both["MZ shown"] = both["MZ"].mask(hide)
    both["diff shown"] = both["diff"].mask(hide)

    # grid covering the study area
    minx, miny, maxx, maxy = boundary.bounds
    xs = np.arange(int(minx // CELL), int(maxx // CELL) + 1)
    ys = np.arange(int(miny // CELL), int(maxy // CELL) + 1)

    def as_grid(col):
        arr = np.full((len(ys), len(xs)), np.nan)
        for (cx, cy), v in both[col].items():
            if xs[0] <= cx <= xs[-1] and ys[0] <= cy <= ys[-1]:
                arr[cy - ys[0], cx - xs[0]] = v
        return arr

    edges_x = np.append(xs, xs[-1] + 1) * CELL
    edges_y = np.append(ys, ys[-1] + 1) * CELL
    vmax = both[["MATSim", "MZ shown"]].max().max()
    dmax = both["diff shown"].abs().max()

    # background lines
    munis = gpd.read_file(SETTLEMENT_FILE).clip(boundary)   # municipality lines only inside the hull
    canton = gpd.read_file(CANTON_FILE)
    canton = canton[canton["KTNR"] == 10]                    # 10 = canton of Fribourg

    share_norm = PowerNorm(gamma=0.5, vmin=0, vmax=vmax)     # square-root stretch: small shares stay visible
    diff_norm = Normalize(vmin=-dmax, vmax=dmax)

    fig, axes = plt.subplots(1, 3, figsize=(ps.TEXT_WIDTH, 3.0))
    panels = [("MATSim", "MATSim", ps.SHARE_CMAP, share_norm, "share of activities [%]", False),
              ("Mikrozensus (weighted)", "MZ shown", ps.SHARE_CMAP, share_norm, "share of activities [%]", True),
              ("Difference MATSim − MZ", "diff shown", ps.DIFF_CMAP, diff_norm, "difference [pp]", True)]
    for ax, (title, col, cmap, norm, unit, mark_hidden) in zip(axes, panels):
        mesh = ax.pcolormesh(edges_x, edges_y, as_grid(col), cmap=cmap, norm=norm)
        if mark_hidden:
            for cx, cy in both.index[hide]:
                ax.add_patch(Rectangle((cx * CELL, cy * CELL), CELL, CELL, facecolor=ps.HIDDEN_FILL,
                                       edgecolor=ps.HIDDEN_EDGE, hatch="////", linewidth=0))
        munis.boundary.plot(ax=ax, color="#C8C8C8", linewidth=0.2)
        canton.boundary.plot(ax=ax, color="black", linewidth=0.5)
        gpd.GeoSeries([boundary]).boundary.plot(ax=ax, color="#555555", linewidth=0.4, linestyle="--")
        ax.plot(*CENTRE, marker="*", color="black", markersize=5)
        ax.set_xlim(edges_x[0], edges_x[-1])
        ax.set_ylim(edges_y[0], edges_y[-1])
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(title)
        fig.colorbar(mesh, ax=ax, orientation="horizontal", fraction=0.05, pad=0.03, label=unit)

    fig.tight_layout()
    ps.save(fig, OUT, "activities_grid")
    return mismatch(p, m), int(hide.sum())


def main():
    ps.apply()
    OUT.mkdir(parents=True, exist_ok=True)
    boundary = load_boundary()
    ms, mz = load("matsim"), load("microcensus")
    print(f"Activities compared: MATSim {len(ms)} | MZ {len(mz)} (from {mz['person_id'].nunique()} respondents)")

    grid_mm, n_hidden = grid_figure(ms, mz, boundary)
    print(f"Grid mismatch (5 km cells, all activities): {grid_mm:.1f} pp ({n_hidden} MZ cells hidden on the map)")

    purpose_groups = [("all activities", lambda a: a)] + [
        (p, lambda a, p=p: a[a["destination_purpose"] == p]) for p in PURPOSES]
    age_groups = [(g, lambda a, g=g: a[a["age_group"] == g]) for g in AGE_LABELS]

    pd.set_option("display.width", 250)
    print("\nShare of activities by distance from the centre, per purpose (%):")
    print(ring_figure(ms, mz, purpose_groups, lambda l: l.capitalize(), "activities_by_purpose").to_string(index=False))
    print("\nShare of activities by distance from the centre, per age group (%):")
    print(ring_figure(ms, mz, age_groups, lambda l: f"Age {l}", "activities_by_age").to_string(index=False))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()