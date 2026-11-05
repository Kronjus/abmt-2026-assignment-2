import pandas as pd

from config import load_boundary
from matsim import load_population, filter_residents

persons, activities, trips = load_population()
persons, activities, trips = filter_residents(persons, activities, trips, load_boundary())

n_trips = trips.groupby("person_id").size()
persons["stays_home"] = persons["person_id"].map(n_trips).fillna(0).eq(0)
persons["missing_home_x"] = persons["home_x"].isna()

print("Rows: home_x missing | Columns: stays at home all day")
print(pd.crosstab(persons["missing_home_x"], persons["stays_home"], margins=True).to_string())