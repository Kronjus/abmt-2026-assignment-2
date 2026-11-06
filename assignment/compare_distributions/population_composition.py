"""Describe the synthetic population: age and sex of the residents, and the shares of the attributes that are
compared with the microcensus in compare.py.

Usage (from the repository root, after assignment/filter_data/filter_data.py and compare.py):
    .venv/Scripts/python assignment/compare_distributions/population_composition.py

Outputs: assignment/compare_distributions/output/population_composition.png and population_composition.csv

Notes
- Residents = persons of the filtered population (home inside the study area), all ages, as in the filter table of
  the report. The age pyramid shows them in 5-year groups, women to the right, as shares of all residents.
- The other two panels use output/comparison.csv (population column, persons aged 6 and older): subscription,
  car availability, PT quality at home and settlement pattern. Employment (right panel, top) is the share of
  all residents with the `employed` attribute.
- Only aggregate shares are plotted.
"""
import gzip
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from activity_chains import OUT, POPULATION

ATTRIBUTES = ("age", "sex", "employed")
PATTERN = re.compile(r'<attribute name="([^"]+)" class="[^"]+">([^<]*)<')
MALE, FEMALE = "#4c78a8", "#e45756"


def read_residents():
    rows, cur = [], None
    with gzip.open(POPULATION, "rt", encoding="utf-8") as f:
        for line in f:
            if "<person " in line:
                cur = {}
            elif cur is not None:
                m = PATTERN.search(line)
                if m and m.group(1) in ATTRIBUTES:
                    cur[m.group(1)] = m.group(2)
                if "</person>" in line:
                    rows.append(cur)
                    cur = None
    df = pd.DataFrame(rows)
    df["age"] = df["age"].astype(int)
    df["employed"] = df["employed"] == "true"
    return df


def pyramid(ax, df):
    edges = list(range(0, 90, 5)) + [200]
    labels = [f"{a}-{a + 4}" for a in range(0, 85, 5)] + ["85+"]
    age_c = pd.cut(df["age"], edges, right=False, labels=labels)
    share = 100 * pd.crosstab(age_c, df["sex"]).reindex(labels) / len(df)
    y = np.arange(len(labels))
    ax.barh(y, -share["m"], color=MALE, label="men")
    ax.barh(y, share["f"], color=FEMALE, label="women")
    ax.set_yticks(y, labels, fontsize=6.5)
    lim = np.ceil(share.to_numpy().max())
    ax.set_xlim(-lim, lim)
    ax.set_xticks(np.arange(-lim, lim + 1, 1), [f"{abs(v):.0f}" for v in np.arange(-lim, lim + 1, 1)], fontsize=7)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("share of residents [%]", fontsize=8)
    ax.set_ylabel("age", fontsize=8)
    ax.set_title("Age and sex", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    return share


def bars(ax, labels, values, title, color="#4c566a"):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=color)
    for yi, v in zip(y, values):
        ax.text(v + 1, yi, f"{v:.0f} %", va="center", fontsize=7)
    ax.set_yticks(y, labels, fontsize=7)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="x", labelsize=7)
    ax.set_title(title, fontsize=8)


def main():
    df = read_residents()
    comp = pd.read_csv(OUT / "comparison.csv")
    pop = comp.set_index(["attribute", "category"])["population_%"]

    fig = plt.figure(figsize=(7.5, 4.6))
    grid = fig.add_gridspec(3, 2, width_ratios=[1.15, 1], hspace=0.95, wspace=0.5)
    ax_pyr = fig.add_subplot(grid[:, 0])
    share = pyramid(ax_pyr, df)

    ax = fig.add_subplot(grid[0, 1])
    persons = [("employed*", 100 * df["employed"].mean()),
               ("car available", pop[("Car availability", "available")]),
               ("PT subscription", 100 - pop[("PT subscription", "none")])]
    bars(ax, [p[0] for p in persons], [p[1] for p in persons], "Persons (* share of all residents)")

    ax = fig.add_subplot(grid[1, 1])
    classes = ["A", "B", "C", "D", "none"]
    bars(ax, ["no class" if c == "none" else f"class {c}" for c in classes],
         [pop[("PT quality at home", c)] for c in classes], "PT quality at home (age 6+)", color="#1b9e77")
    ax.set_ylim(-0.6, 4.6)

    ax = fig.add_subplot(grid[2, 1])
    types = ["urban", "intermediate", "rural"]
    bars(ax, types, [pop[("Settlement pattern", t)] for t in types], "Settlement pattern (age 6+)", color="#d95f02")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "population_composition.png", dpi=200, bbox_inches="tight", pad_inches=0.05)
    out = pd.DataFrame([{"residents": len(df), "mean_age": df["age"].mean(), "women_%": 100 * (df["sex"] == "f").mean(),
                         "employed_%": 100 * df["employed"].mean()}])
    out.to_csv(OUT / "population_composition.csv", index=False, float_format="%.2f")
    print(out.round(1).to_string(index=False))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
