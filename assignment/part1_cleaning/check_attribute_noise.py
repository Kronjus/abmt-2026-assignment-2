"""Bootstrap for the attribute shares (4.2): 95% interval of each MZ share,
and whether the synthetic share lies outside it."""
import numpy as np
import pandas as pd

from config import CLEAN_DIR
from compare_attributes import add_settlement, add_pt_quality_are, shares, ATTRIBUTES

rng = np.random.default_rng(1)
N = 1000

ms = pd.read_csv(CLEAN_DIR / "matsim" / "persons.csv")
ms = ms[ms["in_mz_age_range"]].copy()
mz = pd.read_csv(CLEAN_DIR / "microcensus" / "persons.csv")
for df in (ms, mz):
    add_settlement(df)
    add_pt_quality_are(df)   # 2026 map for both, as in the report

for title, col, cats, kind in ATTRIBUTES:
    p, m = shares(ms, col, cats), shares(mz, col, cats)
    sample = mz.dropna(subset=[col])
    resampled = np.array([shares(sample.sample(len(sample), replace=True,
                                               random_state=rng.integers(1e9)), col, cats).values
                          for _ in range(N)])
    low, high = np.percentile(resampled, [2.5, 97.5], axis=0)
    print(f"\n=== {title}")
    for i, c in enumerate(cats):
        verdict = "OUTSIDE -> real" if not low[i] <= p[c] <= high[i] else "inside -> noise"
        print(f"{c:22s} synthetic {p[c]:5.1f} | MZ {m[c]:5.1f} "
              f"(95% interval {low[i]:5.1f}-{high[i]:5.1f}) | {verdict}")