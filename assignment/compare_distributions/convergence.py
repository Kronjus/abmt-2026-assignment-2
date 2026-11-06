"""Convergence numbers of the simulation run used by the analysis (mode_share.SIM), for methodology.tex.

Usage (from the repository root):
    .venv/Scripts/python assignment/compare_distributions/convergence.py

Prints the numbers quoted in the "Simulation setup" section: the largest change of any mode share from
iteration 15 on and over the last ten iterations (percentage points), and the average executed score.
"""
import pandas as pd

from mode_share import SIM

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
