"""Compare activity chains of the filtered synthetic population with the filtered microcensus,
aggregated and by age group.

A chain is the sequence of activities of one person-day, e.g. H-W-H (home, work, home). Letters:
H home, W work, E education, S shop, L leisure, O other.

Usage (from the repository root, after assignment/filter_data/filter_data.py):
    .venv/Scripts/python assignment/compare_distributions/activity_chains.py

Outputs: assignment/compare_distributions/output/chains.csv, chains.png and chains_mismatch.csv

Notes
- Microcensus shares are weighted with the person weight WP; population shares are unweighted.
- Persons aged under 6 are dropped (the microcensus has no target persons that young).
- Population: activity types are mapped as home*/work*/education*/shop/leisure/other; `*_secondary`
  is folded into its base type. `outside` activities (trips leaving the study area, purpose unknown)
  are skipped. Consecutive identical activities are merged.
- Microcensus: the chain is built from the destination purpose of each trip (f52900 of the last stage
  of each Weg, in the order of the stage number). Start at home; a day without trips is "H"; a day
  that does not end at home is closed with H. Purposes: 2 work, 3 education, 4 shop, 8 leisure, 11 home;
  5 services, 9/10 accompanying -> other; 6 business, 7 service trip -> work. Unknown codes (-99, 1, 12, 13)
  are skipped.
- The mismatch is the total variation distance between the two chain distributions (all chains, not
  only the plotted top ones), in percentage points, as in compare.py.
"""
import gzip
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
POPULATION = DATA / "filtered" / "fribourg_population.xml.gz"
MC = DATA / "filtered" / "microcensus"
OUT = Path(__file__).resolve().parent / "output"

AGE_BINS = [6, 18, 45, 65, 200]
AGE_LABELS = ["6-17", "18-44", "45-64", "65+"]
TOP = 8  # chains shown per panel; the rest is summed into "other chains"

POP_TYPE = {"home": "H", "home_secondary": "H", "work": "W", "work_secondary": "W", "education": "E",
            "education_secondary": "E", "shop": "S", "leisure": "L", "other": "O"}
MC_PURPOSE = {2: "W", 3: "E", 4: "S", 5: "O", 6: "W", 7: "W", 8: "L", 9: "O", 10: "O", 11: "H"}


def merge_repeats(letters):
    out = []
    for a in letters:
        if not out or out[-1] != a:
            out.append(a)
    return "-".join(out)


def read_population():
    rows, person, age, acts = [], None, None, []
    outside = 0

    def flush():
        if person is not None:
            rows.append({"age": age, "chain": merge_repeats(acts)})

    with gzip.open(POPULATION, "rt", encoding="utf-8") as f:
        for line in f:
            if "<person " in line:
                flush()
                person, age, acts = True, None, []
                continue
            if person is None:
                continue
            m = re.search(r'name="age"[^>]*>(\d+)<', line)
            if m:
                age = int(m.group(1))
            m = re.search(r'<activity type="([^"]+)"', line)
            if m:
                if m.group(1) == "outside":
                    outside += 1
                elif not m.group(1).endswith(" interaction"):  # routing stage activities, not real activities
                    acts.append(POP_TYPE[m.group(1)])
        flush()
    print(f"population: {outside} outside activities skipped")
    return pd.DataFrame(rows)


def read_microcensus():
    z = pd.read_csv(MC / "zielpersonen.csv", encoding="latin1", usecols=["HHNR", "alter", "WP"])
    e = pd.read_csv(MC / "etappen.csv", encoding="latin1", usecols=["HHNR", "WEGNR", "ETNR", "f52900"])
    last = e.sort_values("ETNR").groupby(["HHNR", "WEGNR"]).tail(1).sort_values(["HHNR", "ETNR"])
    unknown = ~last["f52900"].isin(MC_PURPOSE)
    print(f"microcensus: {unknown.sum()} trips with unknown purpose skipped")
    last = last[~unknown]
    chains = last.groupby("HHNR")["f52900"].agg(lambda s: merge_repeats(["H"] + [MC_PURPOSE[c] for c in s] + ["H"]))
    z["chain"] = z["HHNR"].map(chains).fillna("H")
    return z.rename(columns={"alter": "age"})


def share(chains, weights=None):
    w = weights if weights is not None else pd.Series(1.0, index=chains.index)
    s = w.groupby(chains).sum()
    return 100 * s / s.sum()


def main():
    pop, mc = read_population(), read_microcensus()
    pop, mc = pop[pop["age"] >= 6].copy(), mc[mc["age"] >= 6].copy()
    for df in (pop, mc):
        df["age_c"] = pd.cut(df["age"], AGE_BINS, right=False, labels=AGE_LABELS)
    print(f"population: {len(pop)} persons; microcensus: {len(mc)} target persons")

    groups = [("All persons", pop, mc)] + [
        (f"Age {a}", pop[pop["age_c"] == a], mc[mc["age_c"] == a]) for a in AGE_LABELS]
    table, summary = [], []
    fig = plt.figure(figsize=(22, 11))
    grid = fig.add_gridspec(2, 4)
    axes = [fig.add_subplot(grid[0, :])] + [fig.add_subplot(grid[1, i]) for i in range(4)]
    for ax, (title, p_df, m_df) in zip(axes, groups):
        p, m = share(p_df["chain"]), share(m_df["chain"], m_df["WP"])
        both = pd.concat([p.rename("population"), m.rename("microcensus")], axis=1).fillna(0.0)
        tvd = 0.5 * (both["population"] - both["microcensus"]).abs().sum()
        summary.append({"group": title, "persons_population": len(p_df), "persons_microcensus": len(m_df),
                        "total_variation_distance_pp": tvd})
        for chain, r in both.iterrows():
            table.append({"group": title, "chain": chain, "population_%": r["population"],
                          "microcensus_%": r["microcensus"], "difference_pp": r["population"] - r["microcensus"]})
        top = both.assign(avg=both.mean(axis=1)).nlargest(TOP, "avg").drop(columns="avg")
        top.loc["other chains"] = both.drop(top.index).sum()
        y = range(len(top))
        ax.barh([i - 0.2 for i in y], top["population"], 0.4, label="population")
        ax.barh([i + 0.2 for i in y], top["microcensus"], 0.4, label="microcensus (weighted)")
        ax.set_yticks(list(y), top.index)
        ax.invert_yaxis()
        ax.set_xlabel("share of persons [%]")
        ax.set_title(f"{title} (n = {len(p_df)} / {len(m_df)})\nmismatch: {tvd:.1f} pp")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", ncol=2, frameon=False)
    fig.text(0.01, 0.012, "Chain letters: H = home, W = work, E = education, S = shop, L = leisure, O = other "
             "(e.g. H-W-H = home, work, home; H alone = stays home all day)", fontsize=10, va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "chains.png", dpi=150)
    df = pd.DataFrame(table).sort_values(["group", "population_%"], ascending=[True, False])
    df.to_csv(OUT / "chains.csv", index=False, float_format="%.2f")
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "chains_mismatch.csv", index=False, float_format="%.2f")
    print(summary.round(1).to_string(index=False))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
