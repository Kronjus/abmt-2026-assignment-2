"""Part 1: ring map - difference in activity shares by distance ring around Fribourg centre,
for all persons and by age group. * = outside the bootstrap range (real difference)."""
import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from shapely.geometry import Point

import plot_style as ps
from config import load_boundary, CRS_LV95
from compare_spatial import load, OUT, CENTRE, RING_EDGES, RING_LABELS, BASEMAP, CANTON_FILE
from compare_chains import AGE_LABELS

N_RESAMPLES = 1000
rng = np.random.default_rng(1)


def ring_shares(acts):
    s = acts.groupby("ring", observed=False)["weight"].sum().reindex(RING_LABELS).fillna(0)
    return 100 * s / s.sum()


def bootstrap_range(acts):
    """95% range of each MZ ring share when respondents are resampled with all their activities."""
    table = (acts.pivot_table(index="person_id", columns="ring", values="weight",
                              aggfunc="sum", observed=False)
             .reindex(columns=RING_LABELS).fillna(0).to_numpy())
    n = len(table)
    results = np.empty((N_RESAMPLES, len(RING_LABELS)))
    for i in range(N_RESAMPLES):
        totals = table[rng.integers(0, n, n)].sum(axis=0)
        results[i] = 100 * totals / totals.sum()
    return np.percentile(results, 2.5, axis=0), np.percentile(results, 97.5, axis=0)


def ring_polygons(boundary):
    """The five distance rings as areas, clipped to the study area."""
    centre = Point(CENTRE)
    polygons = [centre.buffer(outer * 1000).difference(centre.buffer(inner * 1000)).intersection(boundary)
                for inner, outer in zip(RING_EDGES[:-1], RING_EDGES[1:])]
    return gpd.GeoDataFrame({"ring": RING_LABELS}, geometry=polygons, crs=CRS_LV95)


def main():
    ps.apply()
    boundary = load_boundary()
    rings = ring_polygons(boundary)
    canton = gpd.read_file(CANTON_FILE)
    canton = canton[canton["KTNR"] == 10]
    ms, mz = load("matsim"), load("microcensus")

    groups = [("All persons", lambda a: a)] + [
        (f"Age {g}", lambda a, g=g: a[a["age_group"] == g]) for g in AGE_LABELS]

    # difference per ring, and whether it lies outside the bootstrap range
    results = []
    for label, select in groups:
        p, m = ring_shares(select(ms)), ring_shares(select(mz))
        low, high = bootstrap_range(select(mz))
        real = (p.to_numpy() < low) | (p.to_numpy() > high)
        results.append((label, (p - m).to_numpy(), real))

    dmax = max(np.abs(diff).max() for _, diff, _ in results)
    norm = Normalize(vmin=-dmax, vmax=dmax)   # same colour scale for all panels
    minx, miny, maxx, maxy = boundary.bounds

    fig, axes = plt.subplots(2, 3, figsize=(ps.TEXT_WIDTH, 5.4))
    for ax, (label, diff, real) in zip(axes.flat, results):
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_aspect("equal")
        ctx.add_basemap(ax, crs="EPSG:2056", source=BASEMAP, zoom=10, attribution=False, zorder=0)
        rings.assign(diff=diff).plot(ax=ax, column="diff", cmap=ps.DIFF_CMAP, norm=norm, alpha=0.7,
                                     edgecolor="white", linewidth=0.6, zorder=1)
        canton.boundary.plot(ax=ax, color="black", linewidth=0.5, zorder=2)
        gpd.GeoSeries([boundary]).boundary.plot(ax=ax, color="#555555", linewidth=0.4,
                                                linestyle="--", zorder=2)
        ax.plot(*CENTRE, marker="*", color="black", markersize=5, zorder=4)
        for geom, d, is_real in zip(rings.geometry, diff, real):
            pt = geom.representative_point()
            ax.annotate(f"{d:+.1f}{'*' if is_real else ''}", (pt.x, pt.y), ha="center", va="center",
                        fontsize=6.5, fontweight="bold" if is_real else "normal", zorder=5)
        ax.set_axis_off()
        ax.set_title(label)

    for ax in axes.flat[len(results):]:
        ax.set_visible(False)
    mappable = plt.cm.ScalarMappable(cmap=ps.DIFF_CMAP, norm=norm)
    fig.colorbar(mappable, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.04, pad=0.04,
                 label="difference synthetic population - Mikrozensus [percentage points]")
    ps.save(fig, OUT, "activities_ring_map")
    print(f"Wrote {OUT / 'activities_ring_map.pdf'}")


if __name__ == "__main__":
    main()