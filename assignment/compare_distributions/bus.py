"""Check how well the simulation reproduces the bus, the transport mode relevant for the project.

Usage (from the repository root, after assignment/filter_data/filter_data.py):
    .venv/Scripts/python assignment/compare_distributions/bus.py

Outputs: assignment/compare_distributions/output/bus.csv and bus.png

Notes
- The simulation has no bus mode: choice and calibration only know `pt`. A pt leg is a bus leg if the vehicle of
  the leg (vehicle_id of output_legs) ends in `_bus`; other suffixes are rail, ferry, funicular.
- Microcensus: stages (etappen.csv) with f51300 = 10 (Postauto) or 11 (Bus/Schulbus) are bus stages; 12 (tram)
  does not occur in the study area. Rail = 9 (Bahn/Zug), other pt = 14 (Reisecar), 16 (Schiff), 18 (Bergbahn).
- Persons/Wege are those of mode_share.py (home inside the boundary; microcensus Wege weighted with WP). Only
  shares are compared: the simulation is a 10 % sample, the microcensus is weighted to the population.
- Trip level: a pt trip is "bus only" if all its pt legs/stages are bus, "bus + rail" if it mixes, "no bus" otherwise.
  The microcensus pt trips are the Wege with pt main mode (wmittel 2-8), as in mode_share.py.
- Stage distance: simulation leg `distance` (routed, m) vs microcensus rdist (plausibilised, km).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mode_share import MC, MC_MODE, OUT, SIM, read_simulation_trips

BUS_CODES, RAIL_CODES = {10, 11}, {9}
KINDS = ["bus only", "bus + rail", "rail only", "other pt"]
DIST_BINS = [0, 2, 5, 10, 20, np.inf]
DIST_LABELS = ["0-2 km", "2-5 km", "5-10 km", "10-20 km", "20+ km"]


def kind(has_bus, has_rail):
    return np.select([has_bus & ~has_rail, has_bus & has_rail, ~has_bus & has_rail], KINDS[:3], KINDS[3])


def simulation():
    inside = read_simulation_trips()["person"].unique()
    legs = pd.read_csv(SIM / "output_legs.csv.gz", sep=";", usecols=["person", "trip_id", "distance", "mode", "vehicle_id"],
                       dtype={"person": str, "trip_id": str}, low_memory=False)
    legs = legs[legs["person"].isin(inside) & (legs["mode"] == "pt")].copy()
    legs["vehicle"] = legs["vehicle_id"].str.rsplit("_", n=1).str[-1]
    print("simulation pt legs by vehicle:", legs["vehicle"].value_counts().to_dict())
    legs["bus"], legs["rail"] = legs["vehicle"] == "bus", legs["vehicle"] == "rail"
    trips = legs.groupby("trip_id").agg(bus=("bus", "any"), rail=("rail", "any"), n_legs=("bus", "size"))
    trips["kind"] = kind(trips["bus"], trips["rail"])
    n_trips = len(pd.read_csv(SIM / "output_trips.csv.gz", sep=";", usecols=["person", "main_mode"], dtype={"person": str})
                  .query("person in @inside and main_mode in ['car','car_loop','car_passenger','car_passenger_loop','pt','bike','bike_loop','walk','walk_loop','remote_walk']"))
    return legs[legs["bus"]], legs, trips, n_trips


def microcensus():
    w = pd.read_csv(MC / "wege.csv", encoding="latin1", usecols=["HHNR", "WEGNR", "WP", "wmittel"])
    w = w[w["wmittel"].isin(MC_MODE)]
    e = pd.read_csv(MC / "etappen.csv", encoding="latin1", usecols=["HHNR", "WEGNR", "WP", "f51300", "rdist"])
    e = e.merge(w[["HHNR", "WEGNR"]], on=["HHNR", "WEGNR"])
    pt = e[e["f51300"].isin(BUS_CODES | RAIL_CODES | {14, 16, 18})].copy()
    pt["bus"], pt["rail"] = pt["f51300"].isin(BUS_CODES), pt["f51300"].isin(RAIL_CODES)
    trips = pt.groupby(["HHNR", "WEGNR"]).agg(bus=("bus", "any"), rail=("rail", "any"), weight=("WP", "first"))
    trips["kind"] = kind(trips["bus"], trips["rail"])
    # Wege with pt main mode but no pt stage are rare; they only enter the denominator of "all trips"
    return e[e["f51300"].isin(BUS_CODES)], pt, trips, w["WP"].sum(), len(w)


def main():
    s_bus, s_pt, s_trips, s_all = simulation()
    m_bus, m_pt, m_trips, m_all_w, m_all_n = microcensus()

    rows = []
    def add(metric, sim_v, mc_v, sim_n, mc_n):
        rows.append({"metric": metric, "simulation": sim_v, "microcensus": mc_v, "simulation_n": sim_n, "microcensus_n": mc_n})

    s_share = s_trips["kind"].value_counts().reindex(KINDS, fill_value=0)
    m_share = m_trips.groupby("kind")["weight"].sum().reindex(KINDS, fill_value=0)
    for k in KINDS:
        add(f"pt trips: {k} [% of pt trips]", 100 * s_share[k] / s_share.sum(), 100 * m_share[k] / m_share.sum(),
            int(s_share[k]), int((m_trips["kind"] == k).sum()))
    s_any = s_trips["bus"].sum()
    m_any = m_trips.loc[m_trips["bus"], "weight"].sum()
    add("trips with a bus stage [% of all trips]", 100 * s_any / s_all, 100 * m_any / m_all_w, int(s_any), int(m_trips["bus"].sum()))
    add("pt stages that are bus [% of pt stages]", 100 * len(s_bus) / len(s_pt), 100 * m_pt.loc[m_pt["bus"], "WP"].sum() / m_pt["WP"].sum(),
        len(s_bus), len(m_bus))
    s_km, m_km = s_bus["distance"] / 1000, m_bus["rdist"].where(m_bus["rdist"] >= 0)
    add("bus stage length, mean [km]", s_km.mean(), np.average(m_km.dropna(), weights=m_bus.loc[m_km.notna(), "WP"]), len(s_km), int(m_km.notna().sum()))
    add("bus stage length, median [km]", s_km.median(), m_km.median(), len(s_km), int(m_km.notna().sum()))
    add("bus share of pt person-km [%]", 100 * s_bus["distance"].sum() / s_pt["distance"].sum(),
        100 * (m_bus["rdist"].clip(lower=0) * m_bus["WP"]).sum() / (m_pt["rdist"].clip(lower=0) * m_pt["WP"]).sum(), len(s_pt), len(m_pt))
    table = pd.DataFrame(rows)
    table["difference_pp_or_unit"] = table["simulation"] - table["microcensus"]
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "bus.csv", index=False, float_format="%.2f")
    pd.set_option("display.width", 200, "display.max_colwidth", 60)
    print(table.round(2).to_string(index=False))

    # bus stages by distance bin
    s_d = pd.cut(s_km, DIST_BINS, right=False, labels=DIST_LABELS).value_counts(normalize=True).reindex(DIST_LABELS) * 100
    mk = m_bus[m_km.notna()]
    m_d = (pd.Series(mk["WP"].to_numpy()).groupby(pd.cut(m_km.dropna().to_numpy(), DIST_BINS, right=False, labels=DIST_LABELS)).sum()
           .reindex(DIST_LABELS))
    m_d = 100 * m_d / m_d.sum()
    m_n = pd.Series(pd.cut(m_km.dropna().to_numpy(), DIST_BINS, right=False, labels=DIST_LABELS)).value_counts().reindex(DIST_LABELS)
    print("\nbus stage length distribution [%]\n", pd.DataFrame({"sim": s_d, "mc": m_d, "mc_n": m_n}).round(1).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    x, w = np.arange(len(KINDS)), 0.38
    for i, (name, vals) in enumerate((("simulation", [100 * s_share[k] / s_share.sum() for k in KINDS]),
                                      ("microcensus", [100 * m_share[k] / m_share.sum() for k in KINDS]))):
        bars = ax.bar(x + (i - 0.5) * w, vals, w * 0.92, label=name, color=["#4c566a", "#d95f02"][i])
        ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
    ax.set_xticks(x, [f"{k}\nn={int((m_trips['kind'] == k).sum())} mic." for k in KINDS], fontsize=8)
    ax.set_ylabel("share of pt trips [%]")
    ax.set_title("Bus within public transport trips")
    ax.legend(frameon=False)
    ax = axes[1]
    x = np.arange(len(DIST_LABELS))
    for i, (name, vals) in enumerate((("simulation", s_d), ("microcensus", m_d))):
        bars = ax.bar(x + (i - 0.5) * w, vals, w * 0.92, label=name, color=["#4c566a", "#d95f02"][i])
        ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
    ax.set_xticks(x, [f"{l}\nn={m_n[l]} mic." for l in DIST_LABELS], fontsize=8)
    ax.set_ylabel("share of bus stages [%]")
    ax.set_title("Length of bus stages")
    fig.suptitle(f"Bus: simulation ({len(s_bus)} bus legs) vs microcensus ({len(m_bus)} bus stages, unweighted)")
    fig.tight_layout()
    fig.savefig(OUT / "bus.png", dpi=150)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
