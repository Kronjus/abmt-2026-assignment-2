import pandas as pd

from config import load_boundary
from matsim import load_population, filter_residents, handle_outside

persons, activities, trips = load_population()
boundary = load_boundary()

persons, activities, trips = filter_residents(persons, activities, trips, boundary)

# How many residents would the old filter (using home_x) have lost?
print("Residents missing the old 'home_x' attribute:", int(persons["home_x"].isna().sum()))

n_trips = trips.groupby("person_id").size().reindex(persons["person_id"], fill_value=0)
stay_home = int((n_trips == 0).sum())
print(f"Residents with 0 trips (stay at home): {stay_home} ({stay_home / len(persons):.1%})")
print()

# Do 'outside' mode trips and outside-to-outside trips mean the same thing?
both_outside = (trips["origin_type"] == "outside") & (trips["destination_type"] == "outside")
print("Rows: mode is 'outside' | Columns: both ends outside")
print(pd.crosstab(trips["mode"] == "outside", both_outside).to_string())
print()

trips = handle_outside(trips)
print("Modes left:", trips["mode"].value_counts().to_dict())
print("Share of trips crossing the boundary:", round(trips["crosses_boundary"].mean(), 3))