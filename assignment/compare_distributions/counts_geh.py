"""Validate the simulated traffic volumes on the counted links of the network with the GEH statistic.

Usage (from the repository root):
    .venv/Scripts/python assignment/compare_distributions/counts_geh.py

Outputs: assignment/compare_distributions/output/counts_geh.csv (one row per counted link) and counts_geh.png

Notes
- Observed: data/Fribourg_10pct/network_calibration_files/target_counts.csv, 657 links. As used by the network
  calibration (FlowProcessor, CountsProcessor), `count` is the average hourly flow per lane over the 24 h of the day,
  at full scale (veh/h/lane); `weight` is 1.0 for 48 links and 0.3 for the rest.
- Simulated: simulation_output/output_traffic_flow_daily_counts.csv (LinkFlowCounter): vehicles entering the link
  over the day on the 10 % sample scale. Cars and freight count 1 each, buses count 0.1 (full timetable on the
  sample scale), bikes 0. Hourly flow per lane = dailyCount / sampleSize / 24 / lanes, the same normalisation
  the calibration uses.
- GEH = sqrt(2 (M - C)^2 / (M + C)) with M, C the hourly volumes of the link (per-lane value x lanes). The usual
  limits are for hourly volumes: GEH < 5 good, < 10 acceptable. Here the volumes are 24 h averages, so the
  statistic is more lenient than a peak-hour GEH and only the daily level is validated.
- Links with (almost) no simulated traffic (fewer than NEAR_ZERO veh/h) are probably not only an underestimation:
  a count station may be matched to a wrong link or to a link that no simulated vehicle uses in the cut network.
  The statistics are therefore also given without them (counts_geh_summary.csv) and for them alone.
"""
import gzip
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mode_share import DATA, OUT, SIM

COUNTS = DATA / "Fribourg_10pct" / "network_calibration_files" / "target_counts.csv"
SAMPLE_SIZE, HOURS = 0.1, 24.0
NEAR_ZERO = 1.0  # veh/h, below this a link counts as having no simulated traffic


def geh(m, c):
    return np.sqrt(2 * (m - c) ** 2 / (m + c))


def link_attributes(ids):
    rows = {}
    with gzip.open(SIM / "output_network.xml.gz") as f:
        for _, el in ET.iterparse(f, events=("end",)):
            if el.tag == "link" and el.get("id") in ids:
                highway = next((a.text for a in el.iter("attribute") if a.get("name") == "osm:way:highway"), None)
                rows[el.get("id")] = {"lanes": float(el.get("permlanes")), "highway": highway,
                                      "freespeed_kmh": float(el.get("freespeed")) * 3.6}
            if el.tag == "link":
                el.clear()
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("linkId").reset_index()


def main():
    obs = pd.read_csv(COUNTS, dtype={"linkId": str}).rename(columns={"count": "observed_per_lane"})
    sim = pd.read_csv(SIM / "output_traffic_flow_daily_counts.csv", sep=";", dtype={"linkId": str})
    df = obs.merge(sim, on="linkId", how="left").merge(link_attributes(set(obs["linkId"])), on="linkId", how="left")
    print(f"{len(df)} counted links, {df['dailyCount'].isna().sum()} without simulated value, "
          f"{df['lanes'].isna().sum()} not found in the network")
    df["simulated_per_lane"] = df["dailyCount"] / SAMPLE_SIZE / HOURS / df["lanes"]
    df["observed_h"] = df["observed_per_lane"] * df["lanes"]
    df["simulated_h"] = df["simulated_per_lane"] * df["lanes"]
    df["geh"] = geh(df["simulated_h"], df["observed_h"])
    df["rel_error_%"] = 100 * (df["simulated_h"] - df["observed_h"]) / df["observed_h"]
    df.to_csv(OUT / "counts_geh.csv", index=False, float_format="%.3f")

    def summary(d):
        r = np.corrcoef(d["observed_h"], d["simulated_h"])[0, 1] ** 2
        return pd.Series({"links": len(d), "mean_obs_veh_h": d["observed_h"].mean(), "mean_sim_veh_h": d["simulated_h"].mean(),
                          "bias_%": 100 * (d["simulated_h"].sum() / d["observed_h"].sum() - 1),
                          "median_abs_rel_err_%": d["rel_error_%"].abs().median(), "mean_GEH": d["geh"].mean(),
                          "median_GEH": d["geh"].median(), "GEH<5_%": 100 * (d["geh"] < 5).mean(),
                          "GEH<10_%": 100 * (d["geh"] < 10).mean(), "R2": r})
    near_zero = df["simulated_h"] < NEAR_ZERO
    print(f"\n{near_zero.sum()} links with less than {NEAR_ZERO} veh/h simulated, {int(near_zero.sum() - (df['simulated_h'] == 0).sum())} of them "
          f"with a few vehicles; they carry {100 * df.loc[near_zero, 'observed_h'].sum() / df['observed_h'].sum():.1f} % of the observed volume; "
          f"motorway links among them: {int((near_zero & (df['highway'] == 'motorway')).sum())}")
    groups = {"all links": df, "links with simulated traffic": df[~near_zero], "links without simulated traffic": df[near_zero],
              "motorway, with simulated traffic": df[~near_zero & (df["highway"] == "motorway")],
              "weight 1, with simulated traffic": df[~near_zero & (df["weight"] == 1.0)]}
    comparison = pd.DataFrame({name: summary(d) for name, d in groups.items()}).T
    comparison.to_csv(OUT / "counts_geh_summary.csv", index_label="group", float_format="%.2f")
    pd.set_option("display.width", 220)
    print("\nwith and without the links without simulated traffic\n", comparison.round(2).to_string())
    print("\nall links\n", summary(df).round(2).to_string())
    print("\nby weight (1.0 = trusted counts)\n", df.groupby("weight").apply(summary, include_groups=False).round(2).to_string())
    print("\nby road type\n", df.groupby("highway").apply(summary, include_groups=False).round(2).to_string())
    print("\nWorst 10 links\n", df.nlargest(10, "geh")[["linkId", "highway", "lanes", "observed_h", "simulated_h", "geh"]].round(1).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    ok = df["geh"] < 5
    ax.scatter(df.loc[ok, "observed_h"], df.loc[ok, "simulated_h"], s=14, color="#1b9e77", label="GEH < 5")
    ax.scatter(df.loc[~ok, "observed_h"], df.loc[~ok, "simulated_h"], s=14, color="#d95f02", label="GEH >= 5")
    lim = max(df["observed_h"].max(), df["simulated_h"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color="#4c566a", lw=1)
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="observed [veh/h, 24 h average]", ylabel="simulated [veh/h, 24 h average]",
           title="Simulated vs observed volume on counted links")
    ax.legend(frameon=False)
    ax = axes[1]
    ax.hist(df["geh"].clip(upper=20), bins=np.arange(0, 20.5, 1), color="#4c566a")
    ax.axvline(5, color="#d95f02", ls="--")
    ax.axvline(10, color="#d95f02", ls=":")
    ax.set(xlabel="GEH (clipped at 20)", ylabel="links",
           title=f"GEH < 5: {100 * ok.mean():.0f} % of {len(df)} links, GEH < 10: {100 * (df['geh'] < 10).mean():.0f} %")
    fig.tight_layout()
    fig.savefig(OUT / "counts_geh.png", dpi=150, bbox_inches="tight", pad_inches=0.05)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
