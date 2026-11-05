"""MATSim population parsing and cleaning.
The population is synthetic, so printing examples from it is fine."""
import gzip
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from categories import age_class, fold_purpose, CAR_MATSIM

from config import MATSIM_POPULATION, CLEAN_DIR, inside

CACHE_DIR = CLEAN_DIR / "matsim_parsed"


# --- small helpers -----------------------------------------------------------

def parse_time(value):
    """'HH:MM:SS' -> seconds after midnight (may go past 24:00). Missing -> NaN."""
    if value is None or value == "undefined":
        return np.nan
    h, m, s = value.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def convert(value, java_class):
    """Turn the text of an <attribute> into the right Python type."""
    if value is None:
        return None
    if java_class.endswith(("Integer", "Long")):
        return int(value)
    if java_class.endswith("Double"):
        return float(value)
    if java_class.endswith("Boolean"):
        return value == "true"
    return value


def read_attributes(elem):
    """All <attribute> entries directly below an element, as a dictionary."""
    return {a.get("name"): convert(a.text, a.get("class", ""))
            for a in elem.findall("./attributes/attribute")}


def is_stage_activity(activity_type):
    """'car interaction', 'pt interaction', ... are technical, not real activities."""
    return activity_type.endswith(" interaction")


def selected_plan(person):
    plans = person.findall("plan")
    for plan in plans:
        if plan.get("selected") == "yes":
            return plan
    return plans[0] if plans else None


# --- building one trip -------------------------------------------------------

def make_trip(person_id, trip_seq, origin, destination, legs):
    """One trip = all legs between two real activities."""
    first_leg = legs[0] if legs else None
    if first_leg is not None:
        mode = read_attributes(first_leg).get("routingMode", first_leg.get("mode"))
        departure = parse_time(first_leg.get("dep_time"))
    else:
        mode, departure = None, origin["end_time"]

    network_distance = 0.0
    travel_time = 0.0
    for leg in legs:
        route = leg.find("route")
        if route is not None and route.get("distance") is not None:
            network_distance += float(route.get("distance"))
        t = parse_time(leg.get("trav_time"))
        if not np.isnan(t):
            travel_time += t

    return {
        "person_id": person_id,
        "trip_seq": trip_seq,
        "origin_type": origin["type"],
        "destination_type": destination["type"],
        "mode": mode,
        "departure_time": departure,
        "travel_time": travel_time,
        "network_distance": network_distance,
        "crowfly_distance": float(np.hypot(destination["x"] - origin["x"],
                                           destination["y"] - origin["y"])),
        "origin_x": origin["x"], "origin_y": origin["y"],
        "destination_x": destination["x"], "destination_y": destination["y"],
    }


# --- reading the whole file --------------------------------------------------

def parse_population():
    """Read the population file once and return three tables: persons, activities, trips."""
    persons, activities, trips = [], [], []

    with gzip.open(MATSIM_POPULATION, "rb") as f:
        for _, elem in ET.iterparse(f, events=("end",)):
            if elem.tag != "person":
                continue

            person_id = elem.get("id")
            attributes = read_attributes(elem)
            attributes.pop("vehicles", None)  # long technical text we don't need
            persons.append({"person_id": person_id, **attributes})

            plan = selected_plan(elem)
            if plan is not None:
                previous = None   # the last real activity we passed
                legs = []         # legs collected since that activity
                act_seq = 0
                trip_seq = 0

                for item in plan:
                    if item.tag == "leg":
                        legs.append(item)
                    elif item.tag == "activity":
                        if is_stage_activity(item.get("type")):
                            continue
                        activity = {
                            "person_id": person_id,
                            "act_seq": act_seq,
                            "type": item.get("type"),
                            "x": float(item.get("x")),
                            "y": float(item.get("y")),
                            "start_time": parse_time(item.get("start_time")),
                            "end_time": parse_time(item.get("end_time")),
                            "type_before_cutting": read_attributes(item).get("typeBeforeCutting"),
                        }
                        activities.append(activity)
                        act_seq += 1

                        if previous is not None:
                            trips.append(make_trip(person_id, trip_seq, previous, activity, legs))
                            trip_seq += 1
                        previous = activity
                        legs = []

            elem.clear()  # free memory before the next person

    return pd.DataFrame(persons), pd.DataFrame(activities), pd.DataFrame(trips)


def load_population():
    """Parse once, then reuse the saved result (parsing takes a while).
    Delete data/clean/matsim_parsed/ if you change the parser."""
    names = ["persons", "activities", "trips"]
    files = [CACHE_DIR / f"{n}.pkl" for n in names]

    if all(f.exists() for f in files):
        return tuple(pd.read_pickle(f) for f in files)

    tables = parse_population()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for table, f in zip(tables, files):
        table.to_pickle(f)
    return tables

def filter_residents(persons, activities, trips, boundary):
    """Step 4: keep regular agents whose home lies inside the study area."""
    n_all = len(persons)

    is_regular = (persons["subpopulation"].isna()
                  & ~persons["isFreight"].eq(True)
                  & ~persons["isCrossBorder"].eq(True))
    persons = persons[is_regular]
    n_regular = len(persons)

    home_inside = inside(persons["home_coordinate_x"], persons["home_coordinate_y"], boundary)
    persons = persons[home_inside].copy()

    keep = persons["person_id"]
    activities = activities[activities["person_id"].isin(keep)].copy()
    trips = trips[trips["person_id"].isin(keep)].copy()

    print(f"[MATSim step 4] persons: {n_all} -> {n_regular} regular -> {len(persons)} living inside")
    return persons, activities, trips


def handle_outside(trips):
    """Step 5: drop trips entirely outside the area, flag trips that cross the boundary."""
    n_before = len(trips)
    origin_out = trips["origin_type"] == "outside"
    destination_out = trips["destination_type"] == "outside"

    trips = trips[~(origin_out & destination_out)].copy()
    trips["crosses_boundary"] = ((trips["origin_type"] == "outside")
                                 | (trips["destination_type"] == "outside"))

    print(f"[MATSim step 5] trips: {n_before} -> {len(trips)} after removing outside-outside trips")
    return trips

# How MATSim modes map onto the five common modes
MATSIM_MODE_MAP = {
    "car": "car", "car_passenger": "car_passenger", "pt": "pt", "bike": "bike", "walk": "walk",
    "car_loop": "car", "car_passenger_loop": "car_passenger",
    "bike_loop": "bike", "walk_loop": "walk",
    "remote_walk": "walk",
    "outside": "unknown",  # the 131 boundary artefacts
}


def harmonise(persons, activities, trips):
    """Steps 6 + 7: common categories, weights, and trip counts / stay-at-home flags."""
    # --- trips: modes and purposes
    trips = trips.copy()
    unmapped = set(trips["mode"]) - set(MATSIM_MODE_MAP)
    assert not unmapped, f"MATSim modes without a mapping: {unmapped}"
    trips["is_loop"] = trips["mode"].str.endswith("_loop")
    trips["mode"] = trips["mode"].map(MATSIM_MODE_MAP)
    for end in ["origin", "destination"]:
        trips[f"{end}_purpose_detailed"] = trips[f"{end}_type"]
        trips[f"{end}_purpose"] = fold_purpose(trips[f"{end}_type"])

    # --- activities: purposes
    activities = activities.copy()
    activities["purpose_detailed"] = activities["type"]
    activities["purpose"] = fold_purpose(activities["type"])

    # --- persons: age class, weight, trips per person, stay-at-home
    persons = persons.copy()
    persons["age_class"] = age_class(persons["age"])
    persons["in_mz_age_range"] = persons["age"] >= 6   # the Mikrozensus only asks people aged 6+
    persons["weight"] = 1.0                            # every agent counts once
    n_trips = trips.groupby("person_id").size()
    persons["n_trips"] = persons["person_id"].map(n_trips).fillna(0).astype(int)
    persons["is_mobile"] = persons["n_trips"] > 0

    print(f"[MATSim step 6/7] {len(persons)} persons, {len(trips)} trips harmonised")
    return persons, activities, trips

HOUSEHOLDS = MATSIM_POPULATION.parent / "fribourg_households.xml.gz"


def read_household_membership():
    """Dictionary: person id -> household id, from the household file."""
    member_of = {}
    with gzip.open(HOUSEHOLDS, "rb") as f:
        for _, elem in ET.iterparse(f, events=("end",)):
            if elem.tag.endswith("household"):
                for child in elem.iter():
                    if child.tag.endswith("personId"):
                        member_of[child.get("refId")] = elem.get("id")
                elem.clear()
    return member_of


def add_person_attributes(persons):
    """Car availability, PT pass and household type, in the shared categories."""
    persons = persons.copy()
    persons["car_avail"] = persons["carAvail"].map(CAR_MATSIM)
    persons["subscription"] = persons["subscriptions"]

    # Household type, derived from the members present in the population
    persons["household_id"] = persons["person_id"].map(read_household_membership())
    grouped = persons.groupby("household_id")["age"]
    size = grouped.transform("size")
    has_minor = grouped.transform("min") < 18
    age_gap = grouped.transform("max") - grouped.transform("min")
    # Mikrozensus 'with children' includes adult children: so two generations
    # (a minor, or an age gap of 18+ years) count as a household with children.
    has_child = has_minor | (age_gap >= 18)

    kind = pd.Series("other", index=persons.index)
    kind[(size == 2) & ~has_child] = "couple, no children"
    kind[has_child] = "with children"
    kind[size == 1] = "single person"
    persons["household_type"] = kind
    return persons