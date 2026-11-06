"""Bootstrap for the distance-ring shares: how much can the Mikrozensus shares vary by chance?
Respondents are resampled as a whole (with all their activities), 1'000 times."""
import numpy as np

from compare_spatial import load, RING_LABELS, PURPOSES
from compare_chains import AGE_LABELS

N_RESAMPLES = 1000
rng = np.random.default_rng(1)
ms, mz = load("matsim"), load("microcensus")


def ring_shares(acts):
    s = acts.groupby("ring", observed=False)["weight"].sum().reindex(RING_LABELS).fillna(0)
    return 100 * s / s.sum()


def bootstrap_range(acts):
    """95% range of each ring share when respondents are resampled with replacement."""
    # one row per respondent, one column per ring: weighted number of activities
    table = (acts.pivot_table(index="person_id", columns="ring", values="weight",
                              aggfunc="sum", observed=False)
             .reindex(columns=RING_LABELS).fillna(0).to_numpy())
    n = len(table)
    results = np.empty((N_RESAMPLES, len(RING_LABELS)))
    for i in range(N_RESAMPLES):
        totals = table[rng.integers(0, n, n)].sum(axis=0)   # draw n respondents with replacement
        results[i] = 100 * totals / totals.sum()
    return np.percentile(results, 2.5, axis=0), np.percentile(results, 97.5, axis=0)


groups = ([("all activities", lambda a: a)]
          + [(p, lambda a, p=p: a[a["destination_purpose"] == p]) for p in PURPOSES]
          + [(f"age {g}", lambda a, g=g: a[a["age_group"] == g]) for g in AGE_LABELS])

for label, select in groups:
    p, m = ring_shares(select(ms)), ring_shares(select(mz))
    low, high = bootstrap_range(select(mz))
    print(f"\n=== {label}  (MZ respondents: {select(mz)['person_id'].nunique()})")
    print(f"{'ring':6s} {'synthetic':>9s} {'MZ':>6s}   95% range of MZ   verdict")
    for i, r in enumerate(RING_LABELS):
        verdict = "OUTSIDE -> real" if not low[i] <= p[r] <= high[i] else "inside -> noise"
        print(f"{r:6s} {p[r]:9.1f} {m[r]:6.1f}   {low[i]:5.1f} - {high[i]:5.1f}     {verdict}")