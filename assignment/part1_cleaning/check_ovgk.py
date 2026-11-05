from config import load_boundary
from matsim import load_population, filter_residents

persons, activities, trips = load_population()
persons, activities, trips = filter_residents(persons, activities, trips, load_boundary())

print("Person attribute 'ovgk' among residents (%):")
print((persons["ovgk"].value_counts(normalize=True, dropna=False) * 100).round(1).to_string())
