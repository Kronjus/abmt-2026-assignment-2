"""Compare the mode share of the simulation output with the filtered microcensus by the place of residence:
settlement pattern (BFS Stadt/Land-Typologie) and PT quality class (ARE) at the person's home.

Usage (from the repository root, after assignment/filter_data/filter_data.py):
    .venv/Scripts/python assignment/compare_distributions/mode_share_by_region.py

Outputs: assignment/compare_distributions/output/mode_share_by_region.csv and .png

Notes
- Trips, modes and filters are those of mode_share.py. Each trip is assigned the settlement pattern and PT quality
  class of the person's home, not of the trip ends. Both are looked up with the functions of compare.py in the same
  polygons for the simulation (home_x/home_y, LV95) and the microcensus (W_X/W_Y of haushalte.csv, WGS84).
- Persons aged under 6 are dropped, as everywhere.
- Microcensus shares are weighted with WP. Groups with fewer than MIN_N microcensus trips are not drawn
  (marked "n < 30"); they stay in the CSV.
- The mismatch per group is the total variation distance between the two mode share vectors, in percentage points.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compare import PT, SETTLEMENT, pt_quality, settlement_type
from mode_share import MC, MODES, OUT, SIM, read_microcensus_trips, read_simulation_trips

COLORS = dict(zip(MODES, ["#4c566a", "#9aa5b8", "#d95f02", "#1b9e77", "#7570b3"]))
MIN_N = 30
GROUPINGS = [("Settlement pattern", "settlement", SETTLEMENT), ("PT quality class at home", "pt", PT)]


def simulation_trips():
    trips = read_simulation_trips()
    persons = pd.read_csv(SIM / "output_persons.csv.gz", sep=";", dtype={"person": str},
                          usecols=["person", "age", "home_x", "home_y"]).drop_duplicates("person")
    persons["pt"] = pt_quality(persons["home_x"], persons["home_y"], "EPSG:2056")
    persons["settlement"] = settlement_type(persons["home_x"], persons["home_y"], "EPSG:2056")
    return trips.merge(persons[["person", "age", "pt", "settlement"]], on="person")


def microcensus_trips():
    trips = read_microcensus_trips()
    z = pd.read_csv(MC / "zielpersonen.csv", encoding="latin1", usecols=["HHNR", "alter"]).drop_duplicates("HHNR")
    h = pd.read_csv(MC / "haushalte.csv", encoding="latin1", usecols=["HHNR", "W_X", "W_Y"])
    h["pt"] = pt_quality(h["W_X"], h["W_Y"], "EPSG:4326")
    h["settlement"] = settlement_type(h["W_X"], h["W_Y"], "EPSG:4326")
    return trips.merge(z.rename(columns={"alter": "age"}), on="HHNR").merge(h[["HHNR", "pt", "settlement"]], on="HHNR")


def prepare(df):
    return df[df["age"] >= 6].copy()


def mode_shares(df, col, weights=None):
    w = weights if weights is not None else pd.Series(1.0, index=df.index)
    table = w.groupby([df[col], df["mode"]], observed=False).sum().unstack(fill_value=0.0)
    table = table.reindex(columns=MODES, fill_value=0.0)
    return 100 * table.div(table.sum(axis=1), axis=0)


def main():
    sim, mc = prepare(simulation_trips()), prepare(microcensus_trips())
    print(f"trips: simulation {len(sim)}, microcensus {len(mc)}")

    fig, axes = plt.subplots(1, len(GROUPINGS), figsize=(13, 5.5), gridspec_kw={"width_ratios": [3, 5]})
    rows = []
    for ax, (title, col, cats) in zip(axes, GROUPINGS):
        s_share, m_share = mode_shares(sim, col), mode_shares(mc, col, mc["weight"])
        s_n, m_n = sim.groupby(col, observed=False).size(), mc.groupby(col, observed=False).size()
        labels = []
        for k, cat in enumerate(cats):
            enough = m_n.get(cat, 0) >= MIN_N
            have = cat in s_share.index and cat in m_share.index
            mismatch = 0.5 * (s_share.loc[cat] - m_share.loc[cat]).abs().sum() if have and enough else np.nan
            for j, (name, share, n) in enumerate((("sim.", s_share, s_n), ("mic.", m_share, m_n))):
                x = k + (j - 0.5) * 0.4
                count = int(n.get(cat, 0))
                if cat in share.index and (j == 0 or enough):
                    bottom = 0.0
                    for mode in MODES:
                        v = share.loc[cat, mode]
                        ax.bar(x, v, 0.36, bottom=bottom, color=COLORS[mode],
                               label=mode if (title, k, j) == (GROUPINGS[0][0], 0, 0) else None)
                        if v >= 8:
                            ax.text(x, bottom + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white")
                        bottom += v
                else:
                    ax.bar(x, 100, 0.36, color="#e4e4e4", hatch="//", edgecolor="#bbbbbb")
                    ax.text(x, 50, "n < 30", rotation=90, ha="center", va="center", fontsize=8)
                ax.text(x, -2, f"{name}\nn={count}", ha="center", va="top", fontsize=6.5)
            labels.append(f"{cat}\n" + ("mismatch –" if np.isnan(mismatch) else f"mismatch {mismatch:.1f} pp"))
            for name, share, n in (("simulation", s_share, s_n), ("microcensus", m_share, m_n)):
                for mode in MODES:
                    rows.append({"grouping": title, "group": cat, "mode": mode, "source": name,
                                 "share_%": share.loc[cat, mode] if cat in share.index else np.nan,
                                 "trips": int(n.get(cat, 0)), "mismatch_pp": mismatch})
        ax.set_xticks(range(len(cats)), labels, fontsize=8)
        ax.tick_params(axis="x", pad=26)
        ax.set_title(title)
        ax.set_ylim(0, 100)
        ax.set_ylabel("share of trips [%]")
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=len(MODES), frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 0.97))
    fig.suptitle("Mode share by place of residence: simulation (left bars) vs microcensus (right bars)", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "mode_share_by_region.png", dpi=130, bbox_inches="tight", pad_inches=0.05)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "mode_share_by_region.csv", index=False, float_format="%.2f")
    print(df.drop_duplicates(["grouping", "group"])[["grouping", "group", "trips", "mismatch_pp"]].round(1).to_string(index=False))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
