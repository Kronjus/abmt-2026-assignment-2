"""Part 1: compare activity chains, aggregated and by age group (persons aged 6+).
Persons who leave the study area during the day are counted, then excluded,
because their 'outside' activities have no known purpose."""
import matplotlib.pyplot as plt
import pandas as pd

from config import CLEAN_DIR, DATA

OUT = DATA / "results"   # inside data/, so it is not tracked by Git
LETTER = {"home": "H", "work": "W", "education": "E", "shop": "S",
          "leisure": "L", "other": "O", "outside": "X"}
AGE_BINS = [6, 18, 45, 65, 200]          # same groups as the teammate's activity_chains.py
AGE_LABELS = ["6-17", "18-44", "45-64", "65+"]
TOP = 8                                   # chains shown per panel, the rest becomes "other chains"


def load(name):
    p = pd.read_csv(CLEAN_DIR / name / "persons.csv")
    p = p[p["in_mz_age_range"]].copy()
    p["chain"] = p["chain"].map(lambda c: "-".join(LETTER[a] for a in c.split("-")))
    p["age_group"] = pd.cut(p["age"], AGE_BINS, right=False, labels=AGE_LABELS)
    p["leaves_area"] = p["chain"].str.contains("X")
    return p


def shares(df):
    s = df.groupby("chain")["weight"].sum()
    return 100 * s / s.sum()


def compare(p_df, m_df):
    both = pd.concat([shares(p_df).rename("MATSim"),
                      shares(m_df).rename("Mikrozensus")], axis=1).fillna(0.0)
    mismatch = 0.5 * (both["MATSim"] - both["Mikrozensus"]).abs().sum()
    return both, mismatch


def main():
    ms, mz = load("matsim"), load("microcensus")

    for name, df in [("MATSim", ms), ("Mikrozensus", mz)]:
        out_share = 100 * df.loc[df["leaves_area"], "weight"].sum() / df["weight"].sum()
        print(f"{name}: {out_share:.1f}% of persons leave the study area during the day (excluded below)")
    ms, mz = ms[~ms["leaves_area"]], mz[~mz["leaves_area"]]

    groups = [("All persons", ms, mz)] + [
        (f"Age {a}", ms[ms["age_group"] == a], mz[mz["age_group"] == a]) for a in AGE_LABELS]

    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(2, 4)
    axes = [fig.add_subplot(grid[0, :])] + [fig.add_subplot(grid[1, i]) for i in range(4)]
    tables, summary = [], []

    for ax, (title, p_df, m_df) in zip(axes, groups):
        both, mismatch = compare(p_df, m_df)
        summary.append({"group": title, "n MATSim": len(p_df), "n MZ": len(m_df),
                        "mismatch pp": round(mismatch, 1)})
        tables.append(both.assign(group=title))

        top = both.assign(avg=both.mean(axis=1)).nlargest(TOP, "avg").drop(columns="avg")
        top.loc["other chains"] = both.drop(top.index).sum()
        y = range(len(top))
        ax.barh([i - 0.2 for i in y], top["MATSim"], 0.4, label="MATSim")
        ax.barh([i + 0.2 for i in y], top["Mikrozensus"], 0.4, label="Mikrozensus (weighted)")
        ax.set_yticks(list(y), top.index)
        ax.invert_yaxis()
        ax.set_xlabel("share of persons [%]")
        ax.set_title(f"{title} (n = {len(p_df)} / {len(m_df)})\nmismatch: {mismatch:.1f} pp")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", ncol=2, frameon=False)
    fig.text(0.01, 0.01, "H home, W work, E education, S shop, L leisure, O other "
                         "(H alone = stays at home all day)", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "chains.pdf")              # for Overleaf
    fig.savefig(OUT / "chains.png", dpi=150)     # for a quick look
    pd.concat(tables).round(2).to_csv(OUT / "chains.csv", index_label="chain")

    print()
    print(pd.DataFrame(summary).to_string(index=False))
    print("\nTop 10 chains, all persons (%):")
    print(compare(ms, mz)[0].sort_values("Mikrozensus", ascending=False).head(10).round(1).to_string())
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()