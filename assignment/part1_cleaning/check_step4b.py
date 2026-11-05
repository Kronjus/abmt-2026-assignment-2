from config import load_boundary, inside
from matsim import load_population, filter_residents, handle_outside

persons, activities, trips = load_population()
boundary = load_boundary()
persons, activities, trips = filter_residents(persons, activities, trips, boundary)
trips = handle_outside(trips)

odd = trips[trips["mode"] == "outside"]
print("Odd trips by origin -> destination type:")
print(odd.groupby(["origin_type", "destination_type"]).size().to_string())
print()

# Where is the end that is NOT 'outside'?
origin_is_real = odd["origin_type"] != "outside"
real_x = odd["origin_x"].where(origin_is_real, odd["destination_x"])
real_y = odd["origin_y"].where(origin_is_real, odd["destination_y"])
print("Real end lies inside the boundary:", int(inside(real_x, real_y, boundary).sum()), "of", len(odd))
print("Median crow-fly distance (m):", round(float(odd["crowfly_distance"].median())))