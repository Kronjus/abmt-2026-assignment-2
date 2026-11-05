from config import load_boundary
from microcensus import load_persons, add_home_coordinates, filter_home_inside, load_trips

persons = filter_home_inside(add_home_coordinates(load_persons()), load_boundary())
trips = load_trips(persons)

print("Persons with at least one trip:", trips["HHNR"].nunique(), "of", len(persons))
print("Trips per person (unweighted):", round(len(trips) / len(persons), 2))

for col in ["wmittel", "wzweck1", "wzweck2", "pseudo"]:
    print(f"\n{col} codes (number of trips):")
    print(trips[col].value_counts().sort_index().to_string())

print("\nMissing origin coordinates:     ", int(trips["origin_x"].isna().sum()))
print("Missing destination coordinates:", int(trips["destination_x"].isna().sum()))
print("Departure time f51100: min", trips["f51100"].min(), "| max", trips["f51100"].max())
