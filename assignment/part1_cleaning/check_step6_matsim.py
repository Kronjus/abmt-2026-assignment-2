from config import load_boundary
from matsim import load_population, filter_residents, handle_outside, harmonise

persons, activities, trips = load_population()
persons, activities, trips = filter_residents(persons, activities, trips, load_boundary())
trips = handle_outside(trips)
persons, activities, trips = harmonise(persons, activities, trips)

def shares(series):
    return (series.value_counts(normalize=True, dropna=False) * 100).round(1).to_dict()

print("\nMode shares (%):        ", shares(trips["mode"]))
print("Destination purpose (%):", shares(trips["destination_purpose"]))
print("Age classes (%):        ", shares(persons["age_class"].astype(str)))
print("Persons aged 6+:        ", int(persons["in_mz_age_range"].sum()))

adults6 = persons[persons["in_mz_age_range"]]
print(f"Stay at home (6+):       {(~adults6['is_mobile']).mean():.1%}")
print(f"Trips per person (6+):   {adults6['n_trips'].mean():.2f}")
print(f"Trips per mobile person: {adults6.loc[adults6['is_mobile'], 'n_trips'].mean():.2f}")

print("\nOther attributes we could compare (%):")
for col in ["sex", "employed", "carAvail", "bikeAvail", "hasLicense",
            "subscriptions", "municipalityType"]:
    print(f"  {col:17s}", shares(persons[col]))

