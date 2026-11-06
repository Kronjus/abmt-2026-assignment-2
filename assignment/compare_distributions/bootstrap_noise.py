"""Sampling-noise reference for the total variation distances (TVD) of the comparisons.

The filtered microcensus is a small sample, so even a perfect model would show a TVD > 0. We estimate this
noise floor by a person-level bootstrap: draw the microcensus persons with replacement (a person keeps all
their trips, so correlated trips stay together), recompute the weighted distribution and measure its TVD
to the distribution of the full sample. The mean and the 95th percentile of these TVDs are the TVD two
samples of this size would show by chance. The synthetic population is much larger than the microcensus
sample, so its own sampling noise is ignored.

Used by compare.py and mode_share.py (function `tvd_noise`). Run on its own it only prints a small self-test.
"""
import numpy as np
import pandas as pd

N_BOOT = 1000
CHUNK = 100


def tvd_noise(person_counts, n_boot=N_BOOT, seed=0):
    """(mean, 95th percentile) of the bootstrap TVD in percentage points.

    person_counts: DataFrame, one row per person (cluster), one column per category, entries = the
    person's weighted count in that category (e.g. WP for a person in the category, or the summed trip weights).
    Persons without any counts are kept: they still take part in the resampling.
    """
    counts = person_counts.to_numpy(dtype=float)
    full = counts.sum(axis=0)
    full = full / full.sum()
    rng = np.random.default_rng(seed)
    tvds = []
    for start in range(0, n_boot, CHUNK):
        idx = rng.integers(0, len(counts), size=(min(CHUNK, n_boot - start), len(counts)))
        boot = counts[idx].sum(axis=1)
        boot = boot / boot.sum(axis=1, keepdims=True)
        tvds.append(0.5 * np.abs(boot - full).sum(axis=1))
    tvds = 100 * np.concatenate(tvds)
    return float(tvds.mean()), float(np.percentile(tvds, 95))


if __name__ == "__main__":
    demo = pd.DataFrame({"a": [1.0] * 150 + [0.0] * 150, "b": [0.0] * 150 + [1.0] * 150})
    print("self-test (300 persons, 50/50 split):", tvd_noise(demo))
