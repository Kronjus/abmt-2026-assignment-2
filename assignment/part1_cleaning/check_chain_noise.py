"""How much chain mismatch do we get from sample size alone?
Draw MATSim samples of the same size as the MZ group and compare them with all of MATSim."""
import numpy as np
import pandas as pd

from compare_chains import load, AGE_LABELS

ms, mz = load("matsim"), load("microcensus")
ms, mz = ms[~ms["leaves_area"]], mz[~mz["leaves_area"]]


def mismatch(a, b):
    sa = a.value_counts(normalize=True)
    sb = b.value_counts(normalize=True)
    both = pd.concat([sa, sb], axis=1).fillna(0)
    return 50 * (both.iloc[:, 0] - both.iloc[:, 1]).abs().sum()


rng = np.random.default_rng(1)
groups = [("All persons", ms, mz)] + [
    (f"Age {a}", ms[ms["age_group"] == a], mz[mz["age_group"] == a]) for a in AGE_LABELS]

for title, p_df, m_df in groups:
    noise = [mismatch(p_df["chain"].sample(len(m_df), random_state=rng.integers(1e9)), p_df["chain"])
             for _ in range(50)]
    print(f"{title:12s}  n MZ = {len(m_df):4d}   mismatch from sample size alone: "
          f"{np.mean(noise):.1f} pp (range {np.min(noise):.1f}-{np.max(noise):.1f})")
