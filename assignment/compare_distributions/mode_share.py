"""Compare the aggregated mode share of the simulation output with the filtered microcensus and the
target mode shares of the course data.

Usage (from the repository root, after assignment/filter_data/filter_data.py):
    .venv/Scripts/python assignment/compare_distributions/mode_share.py

Outputs: assignment/compare_distributions/output/mode_share.csv and mode_share.png

Notes
- Simulation: output_trips.csv.gz of simulation_output/, only trips of persons whose home lies inside the
  study-area boundary. Modes are folded into five: *_loop -> base mode, remote_walk -> walk. The `outside`
  mode and `truck` (freight) trips are dropped. Trips leaving the study area are dropped too (strict reading
  of "do not use outside activities and modes when validating"): trips starting or ending at an `outside`
  activity, whose far end is only a placeholder of the cutting.
- Microcensus: one record per Weg (wege.csv), weighted with WP, mode = main mode wmittel:
  2-8 -> pt (train, Postauto, ship, tram, bus, other public transport, coach); 9-13 -> car (car, truck,
  taxi, motorbike, moped); 14 -> bike; 15 -> walk. Of the car Wege, those whose longest car stage (etappen.csv,
  f51300: 7 driver, 8 passenger) is as passenger are car_passenger. Plane, device-like vehicles, other and
  pseudo-stages (1, 16, 17, -99) are dropped. Wege whose origin (S_X/S_Y of the first stage) or destination
  (Z_X/Z_Y of the last stage) lies outside the boundary are dropped, as in the simulation.
- Target: data/Fribourg_10pct/global_target_mode_shares.csv.
- The mismatch is the total variation distance between the share vectors, in percentage points, as in compare.py.
  The sampling noise of the microcensus (bootstrap over persons, see bootstrap_noise.py) is shown next to it and
  written to mode_share_mismatch.csv.
"""
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import points

from bootstrap_noise import tvd_noise

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


def boundary_hull():
    return gpd.read_file(BOUNDARY).set_crs("EPSG:2056", allow_override=True).geometry.union_all()


def microcensus_wege_leaving():
    """(HHNR, WEGNR) index of the Wege whose origin or destination lies outside the boundary.
    Missing coordinates (codes below -90) are not counted as leaving."""
    hull = boundary_hull()
    e = pd.read_csv(MC / "etappen.csv", encoding="latin1", usecols=["HHNR", "WEGNR", "ETNR", "S_X", "S_Y", "Z_X", "Z_Y"])
    e = e.sort_values("ETNR").groupby(["HHNR", "WEGNR"])
    first, last = e.head(1).set_index(["HHNR", "WEGNR"]), e.tail(1).set_index(["HHNR", "WEGNR"])
    to_lv95 = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True).transform

    def outside(df, cx, cy):
        x, y = to_lv95(df[cx].to_numpy(), df[cy].to_numpy())
        known = (df[cx].to_numpy() > -90) & (df[cy].to_numpy() > -90)
        return pd.Series(known & ~hull.covers(points(np.asarray(x), np.asarray(y))), index=df.index)

    leaving = outside(first, "S_X", "S_Y") | outside(last, "Z_X", "Z_Y").reindex(first.index, fill_value=False)
    return leaving[leaving].index


def read_simulation_trips():
    hull = boundary_hull()
    persons = pd.read_csv(SIM / "output_persons.csv.gz", sep=";", usecols=["person", "home_x", "home_y"], dtype={"person": str})
    persons = persons[hull.covers(points(persons["home_x"].to_numpy(), persons["home_y"].to_numpy()))]
    trips = pd.read_csv(SIM / "output_trips.csv.gz", sep=";", usecols=["person", "trip_id", "main_mode", "euclidean_distance", "start_activity_type", "end_activity_type"],
                         dtype={"person": str, "trip_id": str})
    trips = trips[trips["person"].isin(persons["person"])]
    leaving = (trips["start_activity_type"] == "outside") | (trips["end_activity_type"] == "outside")
    trips = trips[~leaving & (trips["main_mode"] != "outside")]
    mode = trips["main_mode"].map(SIM_MODE)
    print(f"simulation: {len(persons)} persons inside the boundary, {len(trips)} trips; "
          f"{int(leaving.sum())} trips starting/ending at an outside activity (incl. outside mode) dropped; further dropped {mode.isna().sum()} ({trips.loc[mode.isna(), 'main_mode'].value_counts().to_dict()})")
    trips = trips.assign(mode=mode, distance_km=trips["euclidean_distance"] / 1000)
    return trips.dropna(subset=["mode"])[["person", "trip_id", "mode", "distance_km"]]


def read_simulation():
    return read_simulation_trips()["mode"].value_counts().reindex(MODES, fill_value=0).astype(float)


def read_microcensus_trips():
    w = pd.read_csv(MC / "wege.csv", encoding="latin1", usecols=["HHNR", "WEGNR", "WP", "wmittel", "w_dist_obj2"])
    e = pd.read_csv(MC / "etappen.csv", encoding="latin1", usecols=["HHNR", "WEGNR", "f51300", "rdist"])
    car = e[e["f51300"].isin([7, 8])].sort_values("rdist").groupby(["HHNR", "WEGNR"]).tail(1)
    passenger = car[car["f51300"] == 8].set_index(["HHNR", "WEGNR"]).index
    leaving = pd.MultiIndex.from_frame(w[["HHNR", "WEGNR"]]).isin(microcensus_wege_leaving())
    print(f"microcensus: {leaving.sum()} of {len(w)} Wege leaving the boundary dropped")
    w = w[~leaving].copy()
    w["mode"] = w["wmittel"].map(MC_MODE)
    is_passenger = pd.MultiIndex.from_frame(w[["HHNR", "WEGNR"]]).isin(passenger)
    w.loc[(w["mode"] == "car") & is_passenger, "mode"] = "car_passenger"
    print(f"microcensus: {len(w)} Wege, dropped {w['mode'].isna().sum()} with mode codes "
          f"{sorted(w.loc[w['mode'].isna(), 'wmittel'].unique())}")
    return w.dropna(subset=["mode"]).rename(columns={"WP": "weight", "w_dist_obj2": "distance_km"})[
        ["HHNR", "mode", "distance_km", "weight"]]


def read_microcensus():
    return read_microcensus_trips().groupby("mode")["weight"].sum().reindex(MODES, fill_value=0.0)


def microcensus_noise():
    trips = read_microcensus_trips()
    counts = trips.pivot_table(index="HHNR", columns="mode", values="weight", aggfunc="sum", fill_value=0.0)
    return tvd_noise(counts.reindex(columns=MODES, fill_value=0.0))


def main():
    shares = pd.DataFrame({
        "simulation": read_simulation(),
        "microcensus": read_microcensus(),
        "target": pd.read_csv(TARGET).set_index("mode")["mode_share"].reindex(MODES),
    })
    shares = 100 * shares / shares.sum()
    mismatch = {c: 0.5 * (shares["simulation"] - shares[c]).abs().sum() for c in ("microcensus", "target")}

    noise_mean, noise_p95 = microcensus_noise()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(MODES))
    for i, col in enumerate(shares.columns):
        bars = ax.bar(x + (i - 1) * 0.27, shares[col], 0.27, label=col)
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax.set_xticks(x, MODES)
    ax.set_ylabel("share of trips [%]")
    ax.set_title("Aggregated mode share: simulation vs microcensus vs target\n"
                 f"mismatch of simulation: {mismatch['microcensus']:.1f} pp to microcensus, "
                 f"{mismatch['target']:.1f} pp to target\n"
                 f"sampling noise of the microcensus: {noise_mean:.1f} pp mean, {noise_p95:.1f} pp 95th percentile")
    ax.legend(frameon=False)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "mode_share.png", dpi=150, bbox_inches="tight", pad_inches=0.05)
    shares.to_csv(OUT / "mode_share.csv", index_label="mode", float_format="%.2f")
    print(shares.round(1).to_string())
    print({k: round(v, 1) for k, v in mismatch.items()}, f"noise mean {noise_mean:.1f}, p95 {noise_p95:.1f}")
    pd.DataFrame([{"mismatch_microcensus_pp": mismatch["microcensus"], "mismatch_target_pp": mismatch["target"],
                   "noise_mean_pp": noise_mean, "noise_p95_pp": noise_p95}]).to_csv(
        OUT / "mode_share_mismatch.csv", index=False, float_format="%.2f")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
