import pandas as pd

from config import CLEAN_DIR
from compare_attributes import add_pt_quality_are, shares, PT_CLASSES

ms = pd.read_csv(CLEAN_DIR / "matsim" / "persons.csv")
ms = ms[ms["in_mz_age_range"]].copy()
mz = pd.read_csv(CLEAN_DIR / "microcensus" / "persons.csv")
mz_2015 = shares(mz, "pt_quality", PT_CLASSES)

add_pt_quality_are(ms)
add_pt_quality_are(mz)   # now the MZ homes are classified with the 2026 map too

table = pd.DataFrame({"MATSim (2026 map)": shares(ms, "pt_quality", PT_CLASSES),
                      "MZ (2026 map)": shares(mz, "pt_quality", PT_CLASSES),
                      "MZ (own 2015 class)": mz_2015}).round(1)
print(table.to_string())
for col in ["MZ (2026 map)", "MZ (own 2015 class)"]:
    print(f"Mismatch MATSim vs {col}: {0.5 * (table['MATSim (2026 map)'] - table[col]).abs().sum():.1f} pp")