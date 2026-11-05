"""All category definitions in one place, so both datasets use exactly the same groups."""
import numpy as np
import pandas as pd

# Age classes, following eqasim-switzerland (bounds 6, 15, 18, 24, 45, 65, 80)
AGE_BINS = [0, 6, 15, 18, 24, 45, 65, 80, np.inf]
AGE_LABELS = ["0-5", "6-14", "15-17", "18-23", "24-44", "45-64", "65-79", "80+"]

# The five modes we compare; everything else becomes 'unknown'
MODES = ["car", "car_passenger", "pt", "bike", "walk"]


def age_class(age):
    """Put each age into its class. right=False means 15 belongs to '15-17', not '6-14'."""
    return pd.cut(age, bins=AGE_BINS, labels=AGE_LABELS, right=False)


def fold_purpose(purpose_detailed):
    """'work_secondary' -> 'work', 'home_secondary' -> 'home', etc. Used for activity chains."""
    return purpose_detailed.str.replace("_secondary", "", regex=False)

# --- Mikrozensus codes (following eqasim, checked against the data) ---------

# wmittel: main mode of a trip
MZ_MODE_MAP = {
    1: "pt", 2: "pt", 3: "pt", 4: "pt", 5: "pt", 6: "pt", 7: "pt", 8: "pt", 11: "pt",
    9: "car", 10: "car", 12: "car", 13: "car",
    14: "bike", 16: "bike",
    15: "walk",
    17: "unknown", -99: "unknown",
}
MZ_STAGE_CAR_PASSENGER = 8   # f51300 in etappen.csv: 7 = car driver, 8 = car passenger

# wzweck1: purpose of the activity (trips back home are recognised via wzweck2)
MZ_PURPOSE_MAP = {
    2: "work", 3: "education", 4: "shop", 8: "leisure",
    5: "other", 9: "other", 10: "other",
    6: "work_secondary", 7: "work_secondary",
    11: "home",
    12: "unknown", -99: "unknown", -98: "unknown", -97: "unknown",
}

# --- Person attributes -------------------------------------------------------
# Car availability: MATSim only knows always/never, so "by arrangement" counts as available.
# MZ -99 = younger than 18 or no driving licence -> never. -97/-98 (unknown) stay missing.
CAR_MZ = {1: "available", 2: "available", 3: "never", -99: "never"}
CAR_MATSIM = {"always": "available", "never": "never"}
CAR_CLASSES = ["available", "never"]

# Household type (hhtyp). 230 = single parent, counted as "with children". -97/-98 stay missing.
HH_TYPE_MZ = {10: "single person", 210: "couple, no children",
              220: "with children", 230: "with children", 30: "other"}
HH_TYPES = ["single person", "couple, no children", "with children", "other"]

SUBSCRIPTIONS = ["none", "HT", "VA", "VA+HT", "GA"]

def build_chains(trips, persons):
    """Activity chain per person, e.g. 'home-work-home'. Persons without trips get 'home'."""
    trips = trips.sort_values(["person_id", "trip_seq"])
    first = trips.groupby("person_id")["origin_purpose"].first()
    destinations = trips.groupby("person_id")["destination_purpose"].agg(list)
    chains = {pid: "-".join([first[pid]] + dests) for pid, dests in destinations.items()}
    return persons["person_id"].map(chains).fillna("home")