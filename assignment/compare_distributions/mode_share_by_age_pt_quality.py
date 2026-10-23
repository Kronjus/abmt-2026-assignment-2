"""Compare the mode share by age group and public transport quality of the simulation output with the
filtered microcensus.

Usage (from the repository root, after assignment/filter_data/filter_data.py):
    .venv/Scripts/python assignment/compare_distributions/mode_share_by_age_pt_quality.py

Outputs: assignment/compare_distributions/output/mode_share_by_age_pt_quality.csv and .png

Notes
- Trips, modes and filters are those of mode_share.py. Trips are grouped by the person's age and by the
  OeV-Gueteklasse (ARE: A best .. D, "none" = outside every polygon) at the person's home, not at the trip ends.
- The class is looked up in the same polygons for both sources: simulation home_x/home_y (LV95, output_persons),
  microcensus W_X/W_Y of haushalte.csv (WGS84, converted to LV95). Where polygons overlap, the best class wins.
  As a cross-check, the lookup is compared with the `ovgk` attribute of the simulation persons and with the
  microcensus W_OeV_KLASSE.
- Age groups as in activity_chains.py; persons aged under 6 are dropped. Microcensus age is `alter` of zielpersonen.csv.
- Microcensus shares are weighted with WP. Cells with fewer than MIN_N microcensus trips are not drawn
  (marked "n < 30"); they stay in the CSV.
- The mismatch per cell is the total variation distance between the two mode share vectors, in percentage points.
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer

from activity_chains import AGE_BINS, AGE_LABELS
from mode_share import DATA, MC, MODES, OUT, SIM, read_microcensus_trips, read_simulation_trips

GUETEKLASSEN = DATA / "gueteklassen_oev_2026" / "OeV_Gueteklassen_ARE.gpkg"
CLASSES = ["A", "B", "C", "D", "none"]
COLORS = dict(zip(MODES, ["#4c566a", "#9aa5b8", "#d95f02", "#1b9e77", "#7570b3"]))
MIN_N = 30


def pt_class(x, y):
    """Best OeV-Gueteklasse at the LV95 points, "none" outside every polygon."""
    polygons = gpd.read_file(GUETEKLASSEN, layer="OeV_Gueteklassen_ARE")[["KLASSE", "geometry"]]
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y), crs="EPSG:2056")
    best = gpd.sjoin(pts, polygons, how="left", predicate="within").groupby(level=0)["KLASSE"].min()
    return best.reindex(range(len(pts))).fillna("none").to_numpy()


def simulation_trips():
    trips = read_simulation_trips()
    persons = pd.read_csv(SIM / "output_persons.csv.gz", sep=";", dtype={"person": str},
                          usecols=["person", "age", "ovgk", "home_x", "home_y"]).drop_duplicates("person")
    persons["pt_class"] = pt_class(persons["home_x"], persons["home_y"])
    known = persons[persons["ovgk"].isin(["A", "B", "C", "D"]) | persons["ovgk"].isna()]
    print(f"simulation: pt class vs ovgk attribute agree for {(known['pt_class'] == known['ovgk'].fillna('none')).mean():.1%} "
          f"of {len(known)} persons")
    return trips.merge(persons[["person", "age", "pt_class"]], on="person")


def microcensus_trips():
    trips = read_microcensus_trips()
    z = pd.read_csv(MC / "zielpersonen.csv", encoding="latin1", usecols=["HHNR", "alter"]).drop_duplicates("HHNR")
    h = pd.read_csv(MC / "haushalte.csv", encoding="latin1", usecols=["HHNR", "W_X", "W_Y", "W_OeV_KLASSE"])
    h["x"], h["y"] = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True).transform(
        h["W_X"].to_numpy(), h["W_Y"].to_numpy())
    h["pt_class"] = pt_class(h["x"], h["y"])
    print("microcensus: W_OeV_KLASSE (rows) vs pt class (columns), households")
    print(pd.crosstab(h["W_OeV_KLASSE"], h["pt_class"]).to_string())
    return trips.merge(z.rename(columns={"alter": "age"}), on="HHNR").merge(h[["HHNR", "pt_class"]], on="HHNR")


def prepare(df):
    df = df[df["age"] >= 6].copy()
    df["age_c"] = pd.cut(df["age"], AGE_BINS, right=False, labels=AGE_LABELS)
    return df


def mode_shares(df, weights=None):
    w = weights if weights is not None else pd.Series(1.0, index=df.index)
    table = w.groupby([df["pt_class"], df["age_c"], df["mode"]], observed=False).sum().unstack(fill_value=0.0)
    table = table.reindex(columns=MODES, fill_value=0.0)
    return 100 * table.div(table.sum(axis=1), axis=0)


def main():
    sim, mc = prepare(simulation_trips()), prepare(microcensus_trips())
    print(f"trips: simulation {len(sim)}, microcensus {len(mc)}")
    s_share, m_share = mode_shares(sim), mode_shares(mc, mc["weight"])
    s_n = sim.groupby(["pt_class", "age_c"], observed=False).size()
    m_n = mc.groupby(["pt_class", "age_c"], observed=False).size()

    fig, axes = plt.subplots(1, len(CLASSES), figsize=(24, 6.5), sharey=True)
    rows = []
    for ax, cls in zip(axes, CLASSES):
        for k, age in enumerate(AGE_LABELS):
            mismatch = np.nan
            for j, (name, share, n) in enumerate((("sim.", s_share, s_n), ("mic.", m_share, m_n))):
                x = k + (j - 0.5) * 0.4
                count = int(n.get((cls, age), 0))
                if (cls, age) in share.index and (j == 0 or count >= MIN_N):
                    bottom = 0.0
                    for mode in MODES:
                        v = share.loc[(cls, age), mode]
                        ax.bar(x, v, 0.36, bottom=bottom, color=COLORS[mode], label=mode if (k, j) == (0, 0) and cls == "A" else None)
                        if v >= 8:
                            ax.text(x, bottom + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white")
                        bottom += v
                else:
                    ax.bar(x, 100, 0.36, color="#e4e4e4", hatch="//", edgecolor="#bbbbbb")
                    ax.text(x, 50, "n < 30", rotation=90, ha="center", va="center", fontsize=8)
                ax.text(x, -2, f"{name}\nn={count}", ha="center", va="top", fontsize=6.5)
            if (cls, age) in s_share.index and (cls, age) in m_share.index and m_n.get((cls, age), 0) >= MIN_N:
                mismatch = 0.5 * (s_share.loc[(cls, age)] - m_share.loc[(cls, age)]).abs().sum()
            for name, share, n in (("simulation", s_share, s_n), ("microcensus", m_share, m_n)):
                for mode in MODES:
                    rows.append({"pt_class": cls, "age_group": age, "mode": mode, "source": name,
                                 "share_%": share.loc[(cls, age), mode] if (cls, age) in share.index else np.nan,
                                 "trips": int(n.get((cls, age), 0)), "mismatch_pp": mismatch})
        labels = []
        for age in AGE_LABELS:
            m = [r["mismatch_pp"] for r in rows if r["pt_class"] == cls and r["age_group"] == age][0]
            labels.append(f"{age}\n" + ("mismatch –" if np.isnan(m) else f"mismatch {m:.1f} pp"))
        ax.set_xticks(range(len(AGE_LABELS)), labels, fontsize=8)
        ax.tick_params(axis="x", pad=26)
        ax.set_title("no PT quality class" if cls == "none" else f"PT quality class {cls}")
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("share of trips [%]")
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=len(MODES), frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 0.97))
    fig.suptitle("Mode share by age group and PT quality at home: simulation (left bars) vs microcensus (right bars)",
                 y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "mode_share_by_age_pt_quality.png", dpi=130)
    pd.DataFrame(rows).to_csv(OUT / "mode_share_by_age_pt_quality.csv", index=False, float_format="%.2f")
    print(pd.DataFrame({"simulation_n": s_n, "microcensus_n": m_n}).unstack().to_string())
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
