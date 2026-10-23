"""Compare the mode share by trip distance of the simulation output with the filtered microcensus.

Usage (from the repository root, after assignment/filter_data/filter_data.py):
    .venv/Scripts/python assignment/compare_distributions/mode_share_by_distance.py

Outputs: assignment/compare_distributions/output/mode_share_by_distance.csv and .png

Notes
- Trips, modes and filters are those of mode_share.py. Distance is the straight-line distance between
  origin and destination: euclidean_distance of the simulation trips, w_dist_obj2 of the microcensus Wege.
- Microcensus Wege without a valid distance (negative codes) are dropped. Microcensus shares are weighted
  with WP; bar labels show the number of trips (microcensus: unweighted records).
- The mismatch per bin is the total variation distance between the mode shares of the two sources, in
  percentage points.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mode_share import MODES, OUT, read_microcensus_trips, read_simulation_trips

BINS = [0, 1, 2, 5, 10, 20, np.inf]
LABELS = ["0-1 km", "1-2 km", "2-5 km", "5-10 km", "10-20 km", "20+ km"]
COLORS = dict(zip(MODES, ["#4c566a", "#9aa5b8", "#d95f02", "#1b9e77", "#7570b3"]))


def share_by_bin(trips, weights=None):
    w = weights if weights is not None else pd.Series(1.0, index=trips.index)
    table = w.groupby([trips["bin"], trips["mode"]], observed=False).sum().unstack(fill_value=0.0)
    table = table.reindex(index=LABELS, columns=MODES, fill_value=0.0)
    return 100 * table.div(table.sum(axis=1), axis=0).fillna(0.0)


def main():
    sim, mc = read_simulation_trips(), read_microcensus_trips()
    invalid = mc["distance_km"] < 0
    print(f"microcensus: {invalid.sum()} of {len(mc)} Wege without valid distance dropped")
    mc = mc[~invalid]
    for df in (sim, mc):
        df["bin"] = pd.cut(df["distance_km"], BINS, right=False, labels=LABELS)

    sources = {"simulation": (sim, share_by_bin(sim)),
               "microcensus": (mc, share_by_bin(mc, mc["weight"]))}
    counts = {k: df["bin"].value_counts().reindex(LABELS) for k, (df, _) in sources.items()}
    mismatch = 0.5 * (sources["simulation"][1] - sources["microcensus"][1]).abs().sum(axis=1)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(LABELS))
    width = 0.38
    for i, (name, (_, shares)) in enumerate(sources.items()):
        pos = x + (i - 0.5) * width
        bottom = np.zeros(len(LABELS))
        for mode in MODES:
            ax.bar(pos, shares[mode], width * 0.92, bottom=bottom, color=COLORS[mode],
                   label=mode if i == 0 else None)
            for xi, b, v in zip(pos, bottom, shares[mode]):
                if v >= 6:
                    ax.text(xi, b + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=8, color="white")
            bottom += shares[mode].to_numpy()
        for xi, n in zip(pos, counts[name]):
            ax.text(xi, -3, f"{name[:3]}.\nn={n}", ha="center", va="top", fontsize=7.5)
    ax.set_xticks(x, [f"{l}\nmismatch {m:.1f} pp" for l, m in zip(LABELS, mismatch)])
    ax.tick_params(axis="x", pad=34)
    ax.set_ylim(0, 100)
    ax.set_ylabel("share of trips [%]")
    ax.set_title("Mode share by straight-line trip distance: simulation (left bars) vs microcensus (right bars)", pad=34)
    ax.legend(ncol=len(MODES), frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "mode_share_by_distance.png", dpi=150)

    rows = []
    for name, (_, shares) in sources.items():
        for label in LABELS:
            for mode in MODES:
                rows.append({"distance_bin": label, "mode": mode, "source": name,
                             "share_%": shares.loc[label, mode], "trips": counts[name][label],
                             "mismatch_pp": mismatch[label]})
    pd.DataFrame(rows).to_csv(OUT / "mode_share_by_distance.csv", index=False, float_format="%.2f")
    summary = pd.DataFrame({"simulation_n": counts["simulation"], "microcensus_n": counts["microcensus"],
                            "mismatch_pp": mismatch}).round(1)
    print(summary.to_string())
    for name, (_, shares) in sources.items():
        print(f"\n{name} shares [%]\n{shares.round(1).to_string()}")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
