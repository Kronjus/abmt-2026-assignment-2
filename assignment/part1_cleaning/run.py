"""Part 1: run both cleaning pipelines and save the clean tables to data/clean/.
Prints only aggregates."""
import pandas as pd

from config import CLEAN_DIR, load_boundary
from categories import build_chains
import microcensus as mz
import matsim as ms

PERSON_COLS = ["person_id", "weight", "age", "age_class", "in_mz_age_range", "car_avail",
               "household_type", "subscription", "pt_quality",
               "home_x", "home_y", "n_trips", "is_mobile", "chain"]
TRIP_COLS = ["person_id", "trip_seq", "mode", "origin_purpose", "destination_purpose",
             "origin_purpose_detailed", "destination_purpose_detailed", "departure_time",
             "crowfly_distance", "crosses_boundary",
             "origin_x", "origin_y", "destination_x", "destination_y"]


def clean_microcensus(boundary):
    persons = mz.filter_home_inside(mz.add_home_coordinates(mz.load_persons()), boundary)
    persons, trips = mz.harmonise_trips(mz.load_trips(persons), persons, boundary)
    persons = mz.add_person_attributes(persons)
    persons["in_mz_age_range"] = True
    persons["chain"] = build_chains(trips, persons)
    return persons, trips


def clean_matsim(boundary):
    persons, activities, trips = ms.load_population()
    persons, activities, trips = ms.filter_residents(persons, activities, trips, boundary)
    trips = ms.handle_outside(trips)
    persons, activities, trips = ms.harmonise(persons, activities, trips)
    persons = ms.add_person_attributes(persons)
    # every resident has home_coordinate_x/y, unlike home_x/y (the old bug)
    persons["home_x"] = persons["home_coordinate_x"]
    persons["home_y"] = persons["home_coordinate_y"]
    # eqasim's own PT class (only used if the ARE 2026 file is missing)
    persons["pt_quality"] = persons["ovgk"].replace({"None": "none"}).fillna("none")
    persons["chain"] = build_chains(trips, persons)
    return persons, trips


def save(name, persons, trips):
    folder = CLEAN_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    persons[PERSON_COLS].to_csv(folder / "persons.csv", index=False)
    trips[TRIP_COLS].to_csv(folder / "trips.csv", index=False)
    print(f"Saved {len(persons)} persons and {len(trips)} trips to {folder}")


def chain_shares(persons):
    p = persons[persons["in_mz_age_range"]]
    totals = p.groupby("chain")["weight"].sum()
    return 100 * totals / totals.sum()


def main():
    boundary = load_boundary()
    mz_persons, mz_trips = clean_microcensus(boundary)
    ms_persons, ms_trips = clean_matsim(boundary)

    save("microcensus", mz_persons, mz_trips)
    save("matsim", ms_persons, ms_trips)

    table = pd.DataFrame({"MATSim %": chain_shares(ms_persons),
                          "MZ % (weighted)": chain_shares(mz_persons)}).fillna(0)
    table = table.sort_values("MZ % (weighted)", ascending=False)
    print("\nTop 15 activity chains (persons aged 6+):")
    print(table.head(15).round(1).to_string())


if __name__ == "__main__":
    main()