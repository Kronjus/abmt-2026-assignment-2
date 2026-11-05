import numpy as np
import pandas as pd

from config import CLEAN_DIR
from compare_spatial import CENTRE, RING_EDGES, RING_LABELS


def home_rings(name):
    p = pd.read_csv(CLEAN_DIR / name / "persons.csv")
    p = p[p["in_mz_age_range"]]
    dist_km = np.hypot(p["home_x"] - CENTRE[0], p["home_y"] - CENTRE[1]) / 1000
    ring = pd.cut(dist_km, RING_EDGES, right=False, labels=RING_LABELS)
    s = p.groupby(ring, observed=False)["weight"].sum()
    return 100 * s / s.sum()


table = pd.DataFrame({"MATSim %": home_rings("matsim"), "MZ % (weighted)": home_rings("microcensus")})
table["diff pp"] = table["MATSim %"] - table["MZ % (weighted)"]
print("Where people LIVE, by distance from Fribourg centre:")
print(table.round(1).to_string())