"""Compare the aggregated mode share of the simulation output with the filtered microcensus and the
target mode shares of the course data.

Usage (from the repository root, after assignment/filter_data/filter_data.py):
    .venv/Scripts/python assignment/compare_distributions/mode_share.py

Outputs: assignment/compare_distributions/output/mode_share.csv and mode_share.png

Notes
- Simulation: output_trips.csv.gz of simulation_output/, only trips of persons whose home lies inside the
  study-area boundary. Modes are folded into five: *_loop -> base mode, remote_walk -> walk. `outside`
  (trips of persons outside the area) and `truck` (freight) trips are dropped.
- Microcensus: one record per Weg (wege.csv), weighted with WP, mode = main mode wmittel:
  2-8 -> pt (train, Postauto, ship, tram, bus, other public transport, coach); 9-13 -> car (car, truck,
  taxi, motorbike, moped); 14 -> bike; 15 -> walk. Of the car Wege, those whose longest car stage (etappen.csv,
  f51300: 7 driver, 8 passenger) is as passenger are car_passenger. Plane, device-like vehicles, other and
  pseudo-stages (1, 16, 17, -99) are dropped.
- Target: data/Fribourg_10pct/global_target_mode_shares.csv.
- The mismatch is the total variation distance between the share vectors, in percentage points, as in compare.py.
"""
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely import points

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SIM = ROOT / "simulation_output"
MC = DATA / "filtered" / "microcensus"
TARGET = DATA / "Fribourg_10pct" / "global_target_mode_shares.csv"
BOUNDARY = DATA / "Boundary" / "fribourg_convexhull.shp"
OUT = Path(__file__).resolve().parent / "output"

MODES = ["car", "car_passenger", "pt", "bike", "walk"]
SIM_MODE = {"car": "car", "car_loop": "car", "car_passenger": "car_passenger", "car_passenger_loop": "car_passenger",
            "pt": "pt", "bike": "bike", "bike_loop": "bike", "walk": "walk", "walk_loop": "walk",
            "remote_walk": "walk"}
MC_MODE = {**dict.fromkeys(range(2, 9), "pt"), **dict.fromkeys(range(9, 14), "car"), 14: "bike", 15: "walk"}


def read_simulation_trips():
    hull = gpd.read_file(BOUNDARY).set_crs("EPSG:2056", allow_override=True).geometry.union_all()
    persons = pd.read_csv(SIM / "output_persons.csv.gz", sep=";", usecols=["person", "home_x", "home_y"], dtype={"person": str})
    persons = persons[hull.covers(points(persons["home_x"].to_numpy(), persons["home_y"].to_numpy()))]
    trips = pd.read_csv(SIM / "output_trips.csv.gz", sep=";", usecols=["person", "main_mode", "euclidean_distance"], dtype={"person": str})
    trips = trips[trips["person"].isin(persons["person"])]
    mode = trips["main_mode"].map(SIM_MODE)
    print(f"simulation: {len(persons)} persons inside the boundary, {len(trips)} trips; "
          f"dropped {mode.isna().sum()} ({trips.loc[mode.isna(), 'main_mode'].value_counts().to_dict()})")
    trips = trips.assign(mode=mode, distance_km=trips["euclidean_distance"] / 1000)
    return trips.dropna(subset=["mode"])[["person", "mode", "distance_km"]]


def read_simulation():
    return read_simulation_trips()["mode"].value_counts().reindex(MODES, fill_value=0).astype(float)


def read_microcensus_trips():
    w = pd.read_csv(MC / "wege.csv", encoding="latin1", usecols=["HHNR", "WEGNR", "WP", "wmittel", "w_dist_obj2"])
    e = pd.read_csv(MC / "etappen.csv", encoding="latin1", usecols=["HHNR", "WEGNR", "f51300", "rdist"])
    car = e[e["f51300"].isin([7, 8])].sort_values("rdist").groupby(["HHNR", "WEGNR"]).tail(1)
    passenger = car[car["f51300"] == 8].set_index(["HHNR", "WEGNR"]).index
    w["mode"] = w["wmittel"].map(MC_MODE)
    is_passenger = pd.MultiIndex.from_frame(w[["HHNR", "WEGNR"]]).isin(passenger)
    w.loc[(w["mode"] == "car") & is_passenger, "mode"] = "car_passenger"
    print(f"microcensus: {len(w)} Wege, dropped {w['mode'].isna().sum()} with mode codes "
          f"{sorted(w.loc[w['mode'].isna(), 'wmittel'].unique())}")
    return w.dropna(subset=["mode"]).rename(columns={"WP": "weight", "w_dist_obj2": "distance_km"})[
        ["HHNR", "mode", "distance_km", "weight"]]


def read_microcensus():
    return read_microcensus_trips().groupby("mode")["weight"].sum().reindex(MODES, fill_value=0.0)


def main():
    shares = pd.DataFrame({
        "simulation": read_simulation(),
        "microcensus": read_microcensus(),
        "target": pd.read_csv(TARGET).set_index("mode")["mode_share"].reindex(MODES),
    })
    shares = 100 * shares / shares.sum()
    mismatch = {c: 0.5 * (shares["simulation"] - shares[c]).abs().sum() for c in ("microcensus", "target")}

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(MODES))
    for i, col in enumerate(shares.columns):
        bars = ax.bar(x + (i - 1) * 0.27, shares[col], 0.27, label=col)
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax.set_xticks(x, MODES)
    ax.set_ylabel("share of trips [%]")
    ax.set_title("Aggregated mode share: simulation vs microcensus vs target\n"
                 f"mismatch of simulation: {mismatch['microcensus']:.1f} pp to microcensus, "
                 f"{mismatch['target']:.1f} pp to target")
    ax.legend(frameon=False)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "mode_share.png", dpi=150)
    shares.to_csv(OUT / "mode_share.csv", index_label="mode", float_format="%.2f")
    print(shares.round(1).to_string())
    print({k: round(v, 1) for k, v in mismatch.items()})
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
