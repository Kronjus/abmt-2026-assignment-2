import numpy as np

from config import load_boundary
from microcensus import load_persons, add_home_coordinates, filter_home_inside, load_trips, harmonise_trips

boundary = load_boundary()
persons = filter_home_inside(add_home_coordinates(load_persons()), boundary)
trips = load_trips(persons)
persons, trips = harmonise_trips(trips, persons, boundary)

trips = trips.merge(persons[["HHNR", "weight"]], on="HHNR")


def weighted_shares(col):
    totals = trips.groupby(col)["weight"].sum()
    return (totals / totals.sum() * 100).round(1).sort_values(ascending=False).to_dict()


w = persons["weight"]
mobile = persons[persons["is_mobile"]]
print("\nMode shares, weighted (%):        ", weighted_shares("mode"))
print("Destination purpose, weighted (%):", weighted_shares("destination_purpose"))
print(f"Stay at home, weighted:            {(w * ~persons['is_mobile']).sum() / w.sum():.1%}")
print(f"Trips per person, weighted:        {(w * persons['n_trips']).sum() / w.sum():.2f}")
print(f"Trips per mobile person, weighted: {(mobile['weight'] * mobile['n_trips']).sum() / mobile['weight'].sum():.2f}")
print("Share of trips crossing boundary:  ", round(trips["crosses_boundary"].mean(), 3))
print("First trips starting at home:      ",
      round((trips.loc[trips["trip_seq"] == 0, "origin_purpose"] == "home").mean(), 3))

# Does each trip start where the previous one ended? (ignoring 'outside' places)
prev_x = trips.groupby("HHNR")["destination_x"].shift(1)
prev_y = trips.groupby("HHNR")["destination_y"].shift(1)
gap = np.hypot(trips["origin_x"] - prev_x, trips["origin_y"] - prev_y)
inside_only = trips["origin_purpose"] != "outside"
print("Trips not starting where the previous one ended (>100 m):",
      int((gap[inside_only] > 100).sum()), "of", int(gap[inside_only].notna().sum()))