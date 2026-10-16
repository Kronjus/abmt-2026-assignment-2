"""Compare the filtered synthetic population with the filtered microcensus.

Three socio-demographic attributes (age, car availability, household type) and
two spatial ones (PT quality at home, settlement pattern), as shares of persons.

Usage (from the repository root, after assignment/filter_data/filter_data.py):
    .venv/Scripts/python assignment/compare_distributions/compare.py

Outputs: assignment/compare_distributions/output/comparison.csv and comparison.png

Notes
- Microcensus shares are weighted with the person weight WP; population shares are unweighted.
- The microcensus has no target persons under 6, so the youngest age class differs by design.
- Settlement pattern: population municipalityType is mapped onto the Eurostat DEGURBA classes
  (urbancore/urban -> city, suburban -> town/suburb, rural -> rural); this is an approximation.
- Household type: the population has no such attribute, so it is derived from the household
  members (size, and whether a member is under 18).
"""
import gzip
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
POPULATION = DATA / "filtered" / "fribourg_population.xml.gz"
HOUSEHOLDS = DATA / "Fribourg_10pct" / "fribourg_households.xml.gz"
MC = DATA / "filtered" / "microcensus"
OUT = Path(__file__).resolve().parent / "output"

AGE_BINS = [0, 18, 30, 45, 65, 80, 200]
AGE_LABELS = ["0-17", "18-29", "30-44", "45-64", "65-79", "80+"]
CAR = ["always", "by arrangement", "never"]
HH_TYPE = ["single person", "couple, no children", "with children", "other"]
PT = ["A", "B", "C", "D", "none"]
SETTLEMENT = ["city", "town/suburb", "rural"]
ATTRIBUTES = [  # (title, categories, kind)
    ("Age", AGE_LABELS, "socio-demographic"),
    ("Car availability", CAR, "socio-demographic"),
    ("Household type", HH_TYPE, "socio-demographic"),
    ("PT quality at home", PT, "spatial"),
    ("Settlement pattern", SETTLEMENT, "spatial"),
]


def read_population():
    attrs = {"age": "age", "carAvail": "car", "ovgk": "pt", "municipalityType": "settlement"}
    rows = []
    block = None
    with gzip.open(POPULATION, "rt", encoding="utf-8") as f:
        for line in f:
            if block is None:
                m = re.search(r'<person id="([^"]+)"', line)
                if m:
                    block, row = [], {"id": m.group(1)}
            else:
                if "<plan" in line:  # person attributes end where the plans start
                    rows.append(row)
                    block = None
                    continue
                m = re.search(r'name="([^"]+)"[^>]*>([^<]*)<', line)
                if m and m.group(1) in attrs:
                    row[attrs[m.group(1)]] = m.group(2)
    pop = pd.DataFrame(rows).set_index("id")
    pop["age"] = pop["age"].astype(int)
    return pop


def add_household_type(pop):
    text = gzip.open(HOUSEHOLDS, "rt", encoding="utf-8").read()
    member_of = {}
    for hh in re.findall(r"<household .*?</household>", text, re.S):
        hid = re.search(r'<household id="([^"]+)"', hh).group(1)
        for pid in re.findall(r'refId="([^"]+)"', hh):
            member_of[pid] = hid
    pop["hh"] = pop.index.map(member_of)
    grouped = pop.groupby("hh")["age"]
    size = grouped.transform("size")  # members present in the filtered population
    has_child = grouped.transform(lambda a: (a < 18).any())
    kind = pd.Series("other", index=pop.index)
    kind[(size == 2) & ~has_child] = "couple, no children"
    kind[has_child] = "with children"
    kind[size == 1] = "single person"
    pop["hh_type"] = kind
    print(f"population: {pop['hh'].isna().sum()} persons without a household")


def share(series, categories, weights=None):
    w = weights if weights is not None else pd.Series(1.0, index=series.index)
    total = w.groupby(series).sum().reindex(categories).fillna(0.0)
    return 100 * total / total.sum()


def read_microcensus():
    z = pd.read_csv(MC / "zielpersonen.csv", encoding="latin1", usecols=["HHNR", "alter", "f42100e", "WP"])
    h = pd.read_csv(MC / "haushalte.csv", encoding="latin1", usecols=["HHNR", "hhtyp", "W_OeV_KLASSE", "W_DEGURBA"])
    mc = z.merge(h, on="HHNR")
    mc["age_c"] = pd.cut(mc["alter"], AGE_BINS, right=False, labels=AGE_LABELS)
    mc["car"] = mc["f42100e"].map({1: "always", 2: "by arrangement"}).fillna("never")  # 3 and -99 (n/a) = never
    mc["hh_type"] = mc["hhtyp"].map({10: "single person", 210: "couple, no children", 220: "with children",
                                     230: "with children", 30: "other"})
    mc["pt"] = mc["W_OeV_KLASSE"].map({1: "A", 2: "B", 3: "C", 4: "D", 5: "none"})
    mc["settlement"] = mc["W_DEGURBA"].map({1: "city", 2: "town/suburb", 3: "rural"})
    return mc


def main():
    pop = read_population()
    add_household_type(pop)
    pop["age_c"] = pd.cut(pop["age"], AGE_BINS, right=False, labels=AGE_LABELS)
    pop["car"] = pop["car"].map({"always": "always", "arrangement": "by arrangement", "never": "never"})
    pop["pt"] = pop["pt"].str.replace("None", "none")
    pop["settlement"] = pop["settlement"].map({"urbancore": "city", "urban": "city",
                                               "suburban": "town/suburb", "rural": "rural"})
    mc = read_microcensus()
    print(f"population: {len(pop)} persons; microcensus: {len(mc)} target persons")

    columns = {"Age": "age_c", "Car availability": "car", "Household type": "hh_type",
               "PT quality at home": "pt", "Settlement pattern": "settlement"}
    table = []
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    for ax, (title, cats, kind) in zip(axes, ATTRIBUTES):
        col = columns[title]
        assert pop[col].notna().all(), f"unmapped population values in {title}"
        if mc[col].isna().any():  # e.g. refused/unknown household type codes (-97, -98)
            print(f"microcensus: {mc[col].isna().sum()} persons without {title}, excluded from its shares")
        p, m = share(pop[col], cats), share(mc[col], cats, mc["WP"])
        for c in cats:
            table.append({"type": kind, "attribute": title, "category": c, "population_%": p[c],
                          "microcensus_%": m[c], "difference_pp": p[c] - m[c]})
        x = range(len(cats))
        ax.bar([i - 0.2 for i in x], p.values, 0.4, label="population")
        ax.bar([i + 0.2 for i in x], m.values, 0.4, label="microcensus (weighted)")
        ax.set_xticks(list(x), cats, rotation=30, ha="right")
        ax.set_title(f"{title}\n({kind})")
        ax.set_ylabel("share of persons [%]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))  # leave room for the legend below the plots
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "comparison.png", dpi=150)
    df = pd.DataFrame(table)
    df.to_csv(OUT / "comparison.csv", index=False, float_format="%.2f")
    print(df.round(1).to_string(index=False))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
