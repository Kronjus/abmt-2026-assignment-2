"""How much can the Mikrozensus shares vary by chance? (bootstrap, 1'000 resamples)
Prints the 95% range of each MZ share and whether the synthetic value lies outside it."""
import numpy as np

from summary_table import load, wmean
from compare_chains import AGE_LABELS

rng = np.random.default_rng(1)
ms, mz = load("matsim"), load("microcensus")

groups = [("All persons", ms, mz)] + [
    (f"Age {a}", ms[ms["age_group"] == a], mz[mz["age_group"] == a]) for a in AGE_LABELS]

for indicator, column in [("stay at home", "stays_home"), ("leave area", "leaves_area")]:
    print(f"\n=== {indicator} (%)")
    for label, p_df, m_df in groups:
        p_df, m_df = p_df.copy(), m_df.copy()
        p_df["stays_home"], m_df["stays_home"] = ~p_df["is_mobile"], ~m_df["is_mobile"]

        synthetic = 100 * p_df[column].mean()
        observed = 100 * wmean(m_df[column], m_df["weight"])

        resampled = []
        for _ in range(1000):
            sample = m_df.sample(len(m_df), replace=True, random_state=rng.integers(1e9))
            resampled.append(100 * wmean(sample[column], sample["weight"]))
        low, high = np.percentile(resampled, [2.5, 97.5])

        verdict = "OUTSIDE -> real difference" if not low <= synthetic <= high else "inside -> within noise"
        print(f"{label:12s} synthetic {synthetic:5.1f} | MZ {observed:5.1f} "
              f"(95% range {low:4.1f}-{high:4.1f}) | {verdict}")