"""Map of the simulated mode share per municipality (Gemeinde) inside the study area.

Usage (from the repository root, after the simulation run):
    .venv/Scripts/python assignment/compare_distributions/mode_share_map.py

Outputs: assignment/compare_distributions/output/mode_share_map.png and mode_share_map.csv

Notes
- Trips, modes and filters are those of mode_share.py (five modes, no outside trips, only persons living inside
  the boundary). A trip belongs to the municipality of the person's home (home_x/home_y), not of the trip ends,
  as in the other comparison scripts. The map therefore shows how the residents of a municipality travel.
- One panel per mode, each with its own colour scale (shares of walk and car differ too much for a common one),
  so that differences between municipalities stay visible. The study-area share is given in each panel title.
  The sixth panel shows the number of simulated trips per municipality, i.e. how much each share can be trusted.
- Municipalities (BFS Stadt/Land-Typologie file, EPSG:2056) are clipped at the boundary. Municipalities with
  fewer than MIN_TRIPS simulated trips are hatched grey instead of coloured: their share is too noisy.
- Only the simulation is mapped: the microcensus has too few trips per municipality for a meaningful comparison.
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mode_share import BOUNDARY, MODES, OUT, SIM, read_simulation_trips

STADTLAND = BOUNDARY.parents[1] / "stadtland_typologie" / "Boundaries_G1_GDETYP2012_20230101_de.gpkg"
MIN_TRIPS = 100
N_LABELS = 4  # label the municipalities with the most trips
CMAPS = {"car": "Blues", "car_passenger": "Purples", "pt": "Oranges", "bike": "Greens", "walk": "Reds"}


def municipality_shares():
    hull = gpd.read_file(BOUNDARY).set_crs("EPSG:2056", allow_override=True)
    zones = gpd.read_file(STADTLAND, columns=["GDENR", "GDENAME"]).to_crs("EPSG:2056")
    zones = gpd.clip(zones, hull.geometry.union_all())

    trips = read_simulation_trips()
    persons = pd.read_csv(SIM / "output_persons.csv.gz", sep=";", dtype={"person": str},
                          usecols=["person", "home_x", "home_y"]).drop_duplicates("person")
    homes = gpd.GeoDataFrame(persons[["person"]], geometry=gpd.points_from_xy(persons["home_x"], persons["home_y"]),
                             crs="EPSG:2056").sjoin(zones[["GDENR", "geometry"]], how="left", predicate="within")
    homes = homes.drop_duplicates("person")
    trips = trips.merge(homes[["person", "GDENR"]], on="person")
    print(f"trips: {len(trips)}, {int(trips['GDENR'].isna().sum())} without a municipality (home on a border), dropped")

    counts = trips.groupby(["GDENR", "mode"]).size().unstack(fill_value=0).reindex(columns=MODES, fill_value=0)
    shares = 100 * counts.div(counts.sum(axis=1), axis=0)
    shares["trips"] = counts.sum(axis=1)
    overall = 100 * counts.sum() / counts.to_numpy().sum()
    return hull, zones.merge(shares, left_on="GDENR", right_index=True, how="left"), overall


def main():
    hull, zones, overall = municipality_shares()
    zones["trips"] = zones["trips"].fillna(0)
    enough = zones["trips"] >= MIN_TRIPS
    print(f"{len(zones)} municipalities, {int(enough.sum())} with at least {MIN_TRIPS} trips")

    fig, axes = plt.subplots(2, 3, figsize=(15, 11))
    axes = axes.ravel()
    minx, miny, maxx, maxy = hull.total_bounds
    pad = 0.02 * max(maxx - minx, maxy - miny)
    labelled = zones.nlargest(N_LABELS, "trips")
    offsets = {"Fribourg": (0, 2500), "Villars-sur-Glâne": (0, -3500)}  # keep neighbouring labels apart
    panels = [(m, CMAPS[m], f"{m}\nstudy area: {overall[m]:.1f}%", "share of trips [%]") for m in MODES]
    panels.append(("trips", "Greys", f"trips per municipality (all modes)\ntotal: {int(zones['trips'].sum())}",
                   "simulated trips"))
    for ax, (col, cmap, title, label) in zip(axes, panels):
        zones[~enough].plot(ax=ax, color="#e4e4e4", hatch="///", edgecolor="#bbbbbb", linewidth=0.3)
        shown = zones[enough]
        vmax = shown[col].max() if col == "trips" else float(np.ceil(shown[col].max() / 5) * 5)
        shown.plot(ax=ax, column=col, cmap=cmap, edgecolor="white", linewidth=0.4, legend=True,
                   legend_kwds={"shrink": 0.6, "label": label}, vmin=0, vmax=vmax)
        hull.boundary.plot(ax=ax, color="black", linewidth=1.0)
        for _, row in labelled.iterrows():
            p = row.geometry.representative_point()
            dx, dy = offsets.get(row["GDENAME"], (0, 0))
            ax.annotate(row["GDENAME"], (p.x + dx, p.y + dy), ha="center", fontsize=7.5,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
        ax.set_xlim(minx - pad, maxx + pad)
        ax.set_ylim(miny - pad, maxy + pad)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(title)
    fig.suptitle(f"Simulated mode share by municipality of residence (grey hatched: fewer than {MIN_TRIPS} trips)")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "mode_share_map.png", dpi=150, bbox_inches="tight", pad_inches=0.05)
    zones.drop(columns="geometry")[["GDENR", "GDENAME", "trips", *MODES]].sort_values("trips", ascending=False).to_csv(
        OUT / "mode_share_map.csv", index=False, float_format="%.2f")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
