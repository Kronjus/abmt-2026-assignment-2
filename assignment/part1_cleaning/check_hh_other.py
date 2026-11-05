from config import load_boundary
import matsim as ms

persons, activities, trips = ms.load_population()
persons, activities, trips = ms.filter_residents(persons, activities, trips, load_boundary())
persons = ms.add_person_attributes(persons)

persons["hh_size"] = persons.groupby("household_id")["person_id"].transform("size")
hh_age_gap = persons.groupby("household_id")["age"].transform(lambda a: a.max() - a.min())

other = persons[persons["household_type"] == "other"]
print("'other' persons by household size (%):")
print((other["hh_size"].value_counts(normalize=True).sort_index() * 100).round(1).to_string())
print()
print("Share of 'other' persons whose household has an age gap of 18+ years:",
      round((hh_age_gap[other.index] >= 18).mean(), 3))