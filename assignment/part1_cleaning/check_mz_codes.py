import numpy as np

from config import load_boundary
from microcensus import load_persons, add_home_coordinates, filter_home_inside, load_trips, read_mz

persons = filter_home_inside(add_home_coordinates(load_persons()), load_boundary())
trips = load_trips(persons).merge(persons[["HHNR", "home_x", "home_y"]], on="HHNR")

# Does the trip end at home? (destination within 100 m of the home location)
trips["ends_at_home"] = np.hypot(trips["destination_x"] - trips["home_x"],
                                 trips["destination_y"] - trips["home_y"]) < 100

print("Share of trips ending at home, by wzweck2:")
print(trips.groupby("wzweck2")["ends_at_home"].agg(["mean", "size"]).round(3).to_string())
print("\nShare of trips ending at home, by wzweck1:")
print(trips.groupby("wzweck1")["ends_at_home"].agg(["mean", "size"]).round(3).to_string())

# Which stage modes appear inside car trips?
stages = read_mz("etappen", ["HHNR", "WEGNR", "f51300"])
car_trips = trips.loc[trips["wmittel"] == 9, ["HHNR", "WEGNR"]]
car_stages = stages.merge(car_trips, on=["HHNR", "WEGNR"])
print("\nStage mode codes (f51300) inside car trips:")
print(car_stages["f51300"].value_counts().sort_index().to_string())