"""Part 1: trips per person, stay-at-home and leaving the study area,
MATSim vs Mikrozensus (persons aged 6+), in total and by age group."""
import pandas as pd

from config import CLEAN_DIR, DATA
from compare_chains import AGE_BINS, AGE_LABELS

OUT = DATA / "results"
INDICATORS = ["trips/person", "trips/mobile person", "stay at home %", "leave area %"]


def load(name):
    p = pd.read_csv(CLEAN_DIR / name / "persons.csv")
    p = p[p["in_mz_age_range"]].copy()
    p["age_group"] = pd.cut(p["age"], AGE_BINS, right=False, labels=AGE_LABELS)
    p["leaves_area"] = p["chain"].str.contains("outside")
    return p


def wmean(values, weights):
    """Weighted mean (MATSim weights are all 1, so this is a normal mean there)."""
    return (values * weights).sum() / weights.sum()


def indicators(p):
    mobile = p[p["is_mobile"]]
    return {"n": len(p),
            "trips/person": wmean(p["n_trips"], p["weight"]),
            "trips/mobile person": wmean(mobile["n_trips"], mobile["weight"]),
            "stay at home %": 100 * wmean(~p["is_mobile"], p["weight"]),
            "leave area %": 100 * wmean(p["leaves_area"], p["weight"])}


def to_latex(table):
    """Booktabs table: one row per group, MATSim and MZ side by side for each indicator."""
    head = " & ".join(rf"\multicolumn{{2}}{{c}}{{{c}}}" for c in
                      ["Persons", "Trips/person", "Trips/mobile person", r"Stay at home [\%]", r"Leave area [\%]"])
    lines = [r"\begin{tabular}{l rr rr rr rr rr}", r"\toprule",
             rf" & {head} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
             r"Group" + r" & MATSim & MZ" * 5 + r" \\", r"\midrule"]
    for group, r in table.iterrows():
        cells = [f"{r['n MATSim']:.0f}", f"{r['n MZ']:.0f}"]
        for ind in INDICATORS:
            fmt = "{:.2f}" if ind.startswith("trips") else "{:.1f}"
            cells += [fmt.format(r[f"{ind} MATSim"]), fmt.format(r[f"{ind} MZ"])]
        lines.append(f"{group.replace('-', '--')} & " + " & ".join(cells) + r" \\")
        if group == "All persons":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main():
    ms, mz = load("matsim"), load("microcensus")
    groups = [("All persons", ms, mz)] + [
        (f"Age {a}", ms[ms["age_group"] == a], mz[mz["age_group"] == a]) for a in AGE_LABELS]

    rows = {}
    for label, p_df, m_df in groups:
        a, b = indicators(p_df), indicators(m_df)
        rows[label] = {**{f"{k} MATSim": v for k, v in a.items()}, **{f"{k} MZ": v for k, v in b.items()}}
    table = pd.DataFrame(rows).T

    OUT.mkdir(parents=True, exist_ok=True)
    table.round(2).to_csv(OUT / "summary_table.csv", index_label="group")
    (OUT / "summary_table.tex").write_text(to_latex(table))

    pd.set_option("display.width", 220)
    print(table.round(2).to_string())
    print(f"\nWrote {OUT / 'summary_table.csv'} and {OUT / 'summary_table.tex'}")


if __name__ == "__main__":
    main()