import pandas as pd

from config import load_boundary
from categories import AGE_LABELS, CAR_CLASSES, HH_TYPES, SUBSCRIPTIONS
import microcensus as mz
import matsim as ms

boundary = load_boundary()

# Mikrozensus
mz_persons = mz.filter_home_inside(mz.add_home_coordinates(mz.load_persons()), boundary)
mz_persons, _ = mz.harmonise_trips(mz.load_trips(mz_persons), mz_persons, boundary)
mz_persons = mz.add_person_attributes(mz_persons)

# MATSim (only persons aged 6+, like the Mikrozensus)
persons, activities, trips = ms.load_population()
persons, activities, trips = ms.filter_residents(persons, activities, trips, boundary)
trips = ms.handle_outside(trips)
persons, activities, trips = ms.harmonise(persons, activities, trips)
ms_persons = ms.add_person_attributes(persons)
ms_persons = ms_persons[ms_persons["in_mz_age_range"]]


def shares(df, col, categories):
    totals = df.groupby(col, observed=True)["weight"].sum().reindex(categories).fillna(0)
    return 100 * totals / totals.sum()


attributes = [("age_class", AGE_LABELS[1:]), ("car_avail", CAR_CLASSES),
              ("household_type", HH_TYPES), ("subscription", SUBSCRIPTIONS)]

print(f"\nPersons compared: MZ {len(mz_persons)} | MATSim {len(ms_persons)}")
for col, cats in attributes:
    print(f"\n=== {col}   (MZ persons with missing value: {int(mz_persons[col].isna().sum())})")
    table = pd.DataFrame({"MATSim %": shares(ms_persons, col, cats),
                          "MZ % (weighted)": shares(mz_persons, col, cats)})
    table["diff pp"] = table["MATSim %"] - table["MZ % (weighted)"]
    print(table.round(1).to_string())
    print(f"mismatch (total variation distance): {0.5 * table['diff pp'].abs().sum():.1f} pp")