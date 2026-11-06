"""Shared figure style for the report (Part 1, and Part 2 if the teammate uses it too)."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

MATSIM = "#966283"   # mauve
MZ = "#EABDA0"       # peach
LABELS = {"matsim": "Synthetic population", "mz": "Mikrozensus (weighted)"}

DIFF_CMAP = plt.get_cmap("RdBu_r")    # blue = MATSim fewer, white = equal, red = MATSim more
SHARE_CMAP = ListedColormap(DIFF_CMAP(np.linspace(0.5, 1.0, 256)), name="RdBu_r_upper")  # upper half: white -> red
HIDDEN_FILL, HIDDEN_EDGE = "#EEEEEE", "#A0A0A0"   # hatched cells hidden for privacy

TEXT_WIDTH = 6.3   # inches, about 16 cm = full A4 text width


def apply():
    """Call once at the start of a plotting script."""
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8,
        "axes.titlesize": 9, "axes.titlelocation": "left", "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 200,
    })


def bar_grid(ax, axis="y"):
    """Light gridlines behind the bars ('y' for vertical bars, 'x' for horizontal ones)."""
    ax.grid(True, axis=axis, color="#E6E6E6", linewidth=0.6)
    ax.set_axisbelow(True)


def shared_legend(fig, ax, ncol=2):
    """One legend for the whole figure, centred at the bottom."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, 0))


def save(fig, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.pdf")   # for Overleaf
    fig.savefig(folder / f"{name}.png")   # for a quick look
    plt.close(fig)