"""Convergence numbers of the simulation run used by the analysis (mode_share.SIM), for methodology.tex.

Usage (from the repository root):
    .venv/Scripts/python assignment/compare_distributions/convergence.py

Prints the numbers quoted in the "Simulation setup" section: the largest change of any mode share from
iteration 15 on and over the last ten iterations (percentage points), and the average executed score.
Outputs: assignment/compare_distributions/output/convergence.png (mode shares and score per iteration).
The mode shares are those of modestats.csv, with *_loop modes folded into the base modes; `outside` and
`truck` are dropped and the shares renormalised, as in mode_share.py.
"""
import matplotlib.pyplot as plt
import pandas as pd

from mode_share import MODES, OUT, SIM, SIM_MODE

modes = pd.read_csv(SIM / "modestats.csv", sep=";").set_index("iteration")
scores = pd.read_csv(SIM / "scorestats.csv", sep=";").set_index("iteration")
last = modes.index.max()


def spread(frm):
    window = modes.loc[frm:]
    return 100 * (window.max() - window.min()).max()


print(f"run: {SIM}, last iteration {last}")
print(f"max change of a mode share from iteration 15 on: {spread(15):.3f} pp")
print(f"max change of a mode share over the last ten iterations: {spread(last - 9):.3f} pp")
s = scores["avg_executed"]
print(f"avg executed score: iteration 1 {s.loc[1]:.2f}, iteration 10 {s.loc[10]:.2f}, "
      f"last ten iterations {s.loc[last - 9:].min():.2f} to {s.loc[last - 9:].max():.2f}")


folded = modes.T.groupby(SIM_MODE).sum().T.reindex(columns=MODES)
folded = 100 * folded.div(folded.sum(axis=1), axis=0)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
for mode in MODES:
    ax1.plot(folded.index, folded[mode], label=mode)
ax1.set_xlabel("iteration")
ax1.set_ylabel("share of trips [%]")
ax1.set_title("Mode shares per iteration")
ax1.legend(frameon=False, loc="center right")
ax2.plot(s.index, s, color="#4c566a")
ax2.set_xlabel("iteration")
ax2.set_ylabel("average executed score")
ax2.set_title("Average score per iteration")
for ax in (ax1, ax2):
    ax.axvline(last - 29, color="#999999", linestyle=":")
ax1.text(last - 29, ax1.get_ylim()[1], " weights decay", va="top", ha="left", fontsize=8, color="#666666")
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "convergence.png", dpi=150, bbox_inches="tight", pad_inches=0.05)
print(f"Wrote {OUT / 'convergence.png'}")
