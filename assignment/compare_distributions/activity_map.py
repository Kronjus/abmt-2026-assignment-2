"""Compare the spatial distribution of activities of the filtered synthetic population with the filtered
microcensus, aggregated over all persons and by age group, as smooth density maps.

Usage (from the repository root, after assignment/filter_data/filter_data.py):
    .venv/Scripts/python assignment/compare_distributions/activity_map.py

Outputs in assignment/compare_distributions/output/: activity_map.pdf (all activities) and
activity_map_no_home.pdf (home activities excluded), both A4 portrait at 300 dpi, plus
activity_map_mismatch.csv (aggregate numbers only). No cell-level or location-level table is written.

Notes
- An activity is the destination of a trip. Population: every activity of the plan except the first one
  (the start-of-day home, which has no trip leading to it). Microcensus: the destination (Z_X/Z_Y, WGS84)
  of every stage that ends a Weg, with the purpose from f52900. Activity types and purposes are mapped as
  in activity_chains.py. Unknown purposes (microcensus) are skipped. Trips leaving the study area are dropped:
  `outside` activities and activities reached from an `outside` activity (population), Wege whose origin lies
  outside the boundary (microcensus).
- Density map: every person counts once per 100 m cell (a person's repeated activities at the same place,
  e.g. at home, are not added up), the counts are smoothed with a Gaussian kernel of standard deviation
  SIGMA, and the result is normalised to the share of all person-places per km2 of that dataset and group, so
  datasets of different size are comparable. Microcensus counts are weighted with the person weight WP.
  Only the study-area boundary is drawn; nothing outside it is shown.
- Disclosure control (the microcensus holds real respondents with real destination coordinates):
  * The smoothing blurs every person over a radius of about SIGMA, so single places cannot be seen.
  * Because a smooth map can still show a lone person as a bump, the "effective number of persons" at every
    point (the sum over persons of the kernel weight exp(-d^2 / 2 SIGMA^2), unweighted) is computed, and the
    map is hidden wherever it is below MIN_PERSONS in that dataset and group. The difference is shown only
    where both datasets are shown.
  * No locations or values are written to a file. The TVD in the CSV is one aggregate number per map.
  * Residual risk: groups that overlap (all persons vs age groups) could be subtracted. The result is a smoothed
    field for a group that is itself hidden there, so it is blurred and below the disclosure threshold, but
    it is not strictly impossible to infer something about a very small group.
- Persons aged under 6 are dropped, as in activity_chains.py.
- The basemap tiles (swisstopo Landeskarte grau, free geo.admin.ch WMTS) are downloaded when the script runs.
- The mismatch is the total variation distance between the two smoothed densities (normalised over the
  study area), in percentage points. It is much less dominated by sampling sparsity than a cell-by-cell
  comparison, but is not comparable with the TVDs of the attribute distributions in compare.py.
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

from mode_share import microcensus_wege_leaving
from activity_chains import AGE_BINS, AGE_LABELS, MC_PURPOSE, MC, OUT, POP_TYPE, POPULATION, ROOT

BOUNDARY = ROOT / "data" / "Boundary" / "fribourg_convexhull.shp"
RASTER = 100  # m, resolution of the raster that is smoothed and drawn
SIGMA = 3000  # m, standard deviation of the Gaussian kernel
MIN_PERSONS = 3  # the map is hidden where fewer persons than this contribute (kernel-weighted)
BASEMAP = "https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-grau/default/current/3857/{z}/{x}/{y}.jpeg"  # grey, so the colours stay readable


def read_population():
    rows, person, age, first, prev_outside, pid = [], None, None, True, False, -1
    outside = 0
    with gzip.open(POPULATION, "rt", encoding="utf-8") as f:
        for line in f:
            if "<person " in line:
                person, age, first, prev_outside = True, None, True, False
                pid += 1
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
                elif m.group(1) == "outside" or prev_outside:  # trip leaves the area or starts outside
                    outside += 1
                else:
                    rows.append({"pid": pid, "age": age, "type": POP_TYPE[m.group(1)],
                                 "x": float(m.group(2)), "y": float(m.group(3))})
                prev_outside = m.group(1) == "outside"
    print(f"population: {outside} activities of trips leaving the area skipped")
    return pd.DataFrame(rows)


def read_microcensus():
    z = pd.read_csv(MC / "zielpersonen.csv", encoding="latin1", usecols=["HHNR", "alter", "WP"])
    e = pd.read_csv(MC / "etappen.csv", encoding="latin1", usecols=["HHNR", "WEGNR", "ETNR", "f52900", "Z_X", "Z_Y"])
    last = e.sort_values("ETNR").groupby(["HHNR", "WEGNR"]).tail(1)
    unknown = ~last["f52900"].isin(MC_PURPOSE)
    print(f"microcensus: {unknown.sum()} trips with unknown purpose skipped")
    last = last[~unknown].copy()
    leaving = pd.MultiIndex.from_frame(last[["HHNR", "WEGNR"]]).isin(microcensus_wege_leaving())
    print(f"microcensus: {int(leaving.sum())} Wege leaving the boundary skipped")
    last = last[~leaving].copy()
    last["x"], last["y"] = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True).transform(
        last["Z_X"].to_numpy(), last["Z_Y"].to_numpy())
    last["type"] = last["f52900"].map(MC_PURPOSE)
    df = last.merge(z, on="HHNR")
    return df.rename(columns={"alter": "age", "WP": "weight"}).assign(pid=df["HHNR"])[["pid", "age", "type", "x", "y", "weight"]]


def smooth(a, sigma_cells):
    """Gaussian smoothing of a 2D array with zero padding (separable, sum-preserving away from the edges)."""
    r = int(np.ceil(3 * sigma_cells))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_cells) ** 2)
    k /= k.sum()
    a = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, a)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, a)


def density(df, x_edges, y_edges, inside, weights=None):
    """Smoothed density of person-places and the effective number of persons.

    Returns (share, persons): share = share of all person-places per km2 [%/km2] (nan outside the study area),
    persons = sum over persons of the kernel weight at every raster cell (unweighted)."""
    cell = pd.DataFrame({"pid": df["pid"].to_numpy(), "x": df["x"].to_numpy(), "y": df["y"].to_numpy(),
                         "ix": np.digitize(df["x"], x_edges), "iy": np.digitize(df["y"], y_edges)})
    cell["w"] = weights.to_numpy() if weights is not None else 1.0
    cell = cell.drop_duplicates(["pid", "ix", "iy"])  # every person counts once per raster cell
    sigma_cells = SIGMA / RASTER
    w, _, _ = np.histogram2d(cell["x"], cell["y"], bins=[x_edges, y_edges], weights=cell["w"])
    n, _, _ = np.histogram2d(cell["x"], cell["y"], bins=[x_edges, y_edges])
    w, n = w.T, n.T  # rows = y, columns = x
    dens = smooth(w, sigma_cells)
    share = 100 * dens / w.sum() / (RASTER / 1000) ** 2  # % of all person-places per km2
    persons = smooth(n, sigma_cells) * 2 * np.pi * sigma_cells ** 2  # kernel-weighted number of persons
    return np.where(inside, share, np.nan), np.where(inside, persons, np.nan)


def draw(fields, boundary, extent, title, path):
    groups = list(fields)
    fig, axes = plt.subplots(len(groups), 3, figsize=(8.27, 11.69), squeeze=False)  # A4 portrait
    for row, group in zip(axes, groups):
        p, m, p_eff, m_eff, n = fields[group]
        show_p, show_m = p_eff >= MIN_PERSONS, m_eff >= MIN_PERSONS
        p_plot, m_plot = np.where(show_p, p, np.nan), np.where(show_m, m, np.nan)
        diff_plot = np.where(show_p & show_m, p - m, np.nan)
        shown = np.r_[p_plot[show_p], m_plot[show_m]]
        vmax = max(np.percentile(shown, 99) if len(shown) else 0, 1e-9)
        both = diff_plot[~np.isnan(diff_plot)]
        dmax = max(np.percentile(np.abs(both), 99) if len(both) else 0, 1e-9)
        share_pct = {"population": 100 * show_p.sum() / np.isfinite(p).sum(), "microcensus": 100 * show_m.sum() / np.isfinite(p).sum()}
        panels = [(p_plot, "population", "viridis", dict(vmin=0, vmax=vmax), "share of activities [% per km$^2$]"),
                  (m_plot, "microcensus", "viridis", dict(vmin=0, vmax=vmax), "share of activities [% per km$^2$]"),
                  (diff_plot, "pop. - microcensus", "RdBu_r", dict(norm=TwoSlopeNorm(0, -dmax, dmax)), "difference [pp per km$^2$]")]
        for ax, (data, name, cmap, kw, label) in zip(row, panels):
            data = np.ma.masked_invalid(data)
            img = ax.imshow(data, origin="lower", extent=extent, cmap=cmap, alpha=0.85, interpolation="bilinear", zorder=2, **kw)
            cx.add_basemap(ax, crs="EPSG:2056", source=BASEMAP, zoom=12, zorder=1, attribution=False)
            boundary.boundary.plot(ax=ax, color="black", linewidth=0.8, zorder=3)
            if name == "pop. - microcensus":
                sub = f"{100 * (show_p & show_m).sum() / np.isfinite(p).sum():.0f} % of the area shown"
            else:
                sub = f"n = {n[0] if name == 'population' else n[1]}; {share_pct[name]:.0f} % of the area shown"
            ax.set_title(f"{group}: {name}\n({sub})", fontsize=6.5)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            cb = fig.colorbar(img, ax=ax, shrink=0.8, pad=0.02, format="%.2g")
            cb.ax.tick_params(labelsize=5)
            cb.set_label(label, fontsize=5.5)
    fig.suptitle(title, fontsize=8)
    fig.text(0.99, 0.003, "Basemap: \u00a9 swisstopo (Landeskarte grau)", ha="right", fontsize=5)
    fig.tight_layout(rect=(0, 0.008, 1, 0.985), pad=0.4, h_pad=0.3, w_pad=0.3)
    fig.savefig(path, dpi=300)  # fixed A4 page, no cropping
    plt.close(fig)


def run(pop, mc, boundary, x_edges, y_edges, inside, name, title, summary):
    extent = (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1])
    groups = [("All persons", pop, mc)] + [
        (f"Age {a}", pop[pop["age_c"] == a], mc[mc["age_c"] == a]) for a in AGE_LABELS]
    fields = {}
    for title_g, p_df, m_df in groups:
        p, p_eff = density(p_df, x_edges, y_edges, inside)
        m, m_eff = density(m_df, x_edges, y_edges, inside, m_df["weight"])
        fields[title_g] = (p, m, p_eff, m_eff, (len(p_df), len(m_df)))
        # TVD of the two densities, both normalised to sum 1 over the study area
        pn, mn = p[inside] / np.nansum(p[inside]), m[inside] / np.nansum(m[inside])
        summary.append({"figure": name, "group": title_g, "activities_population": len(p_df),
                        "activities_microcensus": len(m_df), "total_variation_distance_pp": 100 * 0.5 * np.abs(pn - mn).sum()})
    draw(fields, boundary, extent, title, OUT / f"{name}.pdf")


def main():
    pop, mc = read_population(), read_microcensus()
    boundary = gpd.read_file(BOUNDARY).set_crs("EPSG:2056", allow_override=True)
    hull = boundary.geometry.union_all()
    minx, miny, maxx, maxy = hull.bounds
    x_edges = np.arange(np.floor(minx / RASTER) * RASTER, maxx + RASTER, RASTER)
    y_edges = np.arange(np.floor(miny / RASTER) * RASTER, maxy + RASTER, RASTER)
    xc, yc = np.meshgrid((x_edges[:-1] + x_edges[1:]) / 2, (y_edges[:-1] + y_edges[1:]) / 2)
    inside = hull.covers(points(xc.ravel(), yc.ravel())).reshape(xc.shape)

    pop, mc = pop[pop["age"] >= 6].copy(), mc[mc["age"] >= 6].copy()
    kept = []
    for label, df in (("population", pop), ("microcensus", mc)):
        ok = pd.Series(hull.covers(points(df["x"].to_numpy(), df["y"].to_numpy())), index=df.index)
        print(f"{label}: {int((~ok).sum())} of {len(df)} activities outside the boundary dropped")
        df = df[ok].copy()
        df["age_c"] = pd.cut(df["age"], AGE_BINS, right=False, labels=AGE_LABELS)
        kept.append(df)
    pop, mc = kept

    summary = []
    run(pop, mc, boundary, x_edges, y_edges, inside, "activity_map",
        f"Spatial density of activities (Gaussian kernel, sigma = {SIGMA} m; hidden where fewer than {MIN_PERSONS} persons contribute), all activity types", summary)
    run(pop[pop["type"] != "H"], mc[mc["type"] != "H"], boundary, x_edges, y_edges, inside, "activity_map_no_home",
        f"Spatial density of activities (Gaussian kernel, sigma = {SIGMA} m; hidden where fewer than {MIN_PERSONS} persons contribute), home activities excluded", summary)

    OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "activity_map_mismatch.csv", index=False, float_format="%.2f")
    print(summary.round(1).to_string(index=False))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
