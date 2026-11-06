"""Compare the mode share of the simulation output with the filtered microcensus, by two socio-demographics:
age group and PT subscription.

Usage (from the repository root, after assignment/filter_data/filter_data.py):
    .venv/Scripts/python assignment/compare_distributions/mode_share_by_sociodemographics.py

Outputs: assignment/compare_distributions/output/mode_share_by_sociodemographics.csv and .png

Notes
- Trips, modes and filters are those of mode_share.py. Each trip is assigned the age group and PT subscription
  of the person.
- Age groups as in activity_chains.py; persons aged under 6 are dropped. Microcensus age is `alter` of zielpersonen.csv,
  simulation age is the `age` attribute of output_persons.
- PT subscription as in compare.py: simulation attribute `subscriptions` (GA, HT, VA, VA+HT, none); microcensus
  GA = f41610a, HT = f41610b, VA = f41610c or f41610d, GA wins, HT + VA gives VA+HT, anything else is none.
- Microcensus shares are weighted with WP. Groups with fewer than MIN_N microcensus trips are not drawn
  (marked "n < 30"); they stay in the CSV.
- The mismatch per group is the total variation distance between the two mode share vectors, in percentage points.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from activity_chains import AGE_BINS, AGE_LABELS
from mode_share import MC, MODES, OUT, SIM, read_microcensus_trips, read_simulation_trips

SUB = ["GA", "HT", "VA", "VA+HT", "none"]
COLORS = dict(zip(MODES, ["#4c566a", "#9aa5b8", "#d95f02", "#1b9e77", "#7570b3"]))
MIN_N = 30
GROUPINGS = [("Age group", "age_c", AGE_LABELS), ("PT subscription", "sub", SUB)]


def simulation_trips():
    trips = read_simulation_trips()
    persons = pd.read_csv(SIM / "output_persons.csv.gz", sep=";", dtype={"person": str},
                          usecols=["person", "age", "subscriptions"]).drop_duplicates("person")
    return trips.merge(persons.rename(columns={"subscriptions": "sub"}), on="person")


def microcensus_trips():
    trips = read_microcensus_trips()
    z = pd.read_csv(MC / "zielpersonen.csv", encoding="latin1",
                    usecols=["HHNR", "alter", "f41610a", "f41610b", "f41610c", "f41610d"]).drop_duplicates("HHNR")
    ga, ht = z["f41610a"] == 1, z["f41610b"] == 1
    va = (z["f41610c"] == 1) | (z["f41610d"] == 1)
    z["sub"] = np.select([ga, ht & va, ht, va], ["GA", "VA+HT", "HT", "VA"], "none")
    return trips.merge(z[["HHNR", "alter", "sub"]].rename(columns={"alter": "age"}), on="HHNR")


def prepare(df):
    df = df[df["age"] >= 6].copy()
    df["age_c"] = pd.cut(df["age"], AGE_BINS, right=False, labels=AGE_LABELS)
    return df


def mode_shares(df, col, weights=None):
    w = weights if weights is not None else pd.Series(1.0, index=df.index)
    table = w.groupby([df[col], df["mode"]], observed=False).sum().unstack(fill_value=0.0)
    table = table.reindex(columns=MODES, fill_value=0.0)
    return 100 * table.div(table.sum(axis=1), axis=0)


def main():
    sim, mc = prepare(simulation_trips()), prepare(microcensus_trips())
    print(f"trips: simulation {len(sim)}, microcensus {len(mc)}")
    unknown = sim["sub"].isna().sum()
    if unknown:
        print(f"simulation: {unknown} trips of persons without a subscription attribute dropped from the PT subscription view")

    fig, axes = plt.subplots(1, len(GROUPINGS), figsize=(14, 5.5), gridspec_kw={"width_ratios": [6, 5]})
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
    fig.suptitle("Mode share by age group and PT subscription: simulation (left bars) vs microcensus (right bars)", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "mode_share_by_sociodemographics.png", dpi=130, bbox_inches="tight", pad_inches=0.05)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "mode_share_by_sociodemographics.csv", index=False, float_format="%.2f")
    print(df.drop_duplicates(["grouping", "group"])[["grouping", "group", "trips", "mismatch_pp"]].round(1).to_string(index=False))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
