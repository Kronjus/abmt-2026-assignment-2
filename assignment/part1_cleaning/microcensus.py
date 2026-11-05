"""Mikrozensus (MZMV 2015) cleaning.
Rule: nothing in this file ever prints individual rows, only counts and shares."""
import numpy as np
import pandas as pd
from pyproj import Transformer

from config import MZ_DIR, CRS_LV03, CRS_LV95, inside
from categories import (MZ_MODE_MAP, MZ_STAGE_CAR_PASSENGER, MZ_PURPOSE_MAP, fold_purpose,
                        age_class, CAR_MZ, HH_TYPE_MZ)

WEEKDAYS = [1, 2, 3, 4, 5]  # column 'tag': 1 = Monday


def read_mz(name, columns):
    """Read only the listed columns of one Mikrozensus file (e.g. 'zielpersonen')."""
    return pd.read_csv(MZ_DIR / f"{name}.csv", usecols=columns, encoding="latin-1")


def load_persons():
    """Step 1: load the target persons and keep only Monday-Friday reporting days."""
    persons = read_mz("zielpersonen", ["HHNR", "tag", "WP"])
    n_before = len(persons)

    persons = persons[persons["tag"].isin(WEEKDAYS)].copy()

    print(f"[MZ step 1] persons: {n_before} -> {len(persons)} after weekday filter")
    return persons

def add_home_coordinates(persons):
    """Step 2a: attach each person's home location (from haushalte.csv) in LV95."""
    households = read_mz("haushalte", ["HHNR", "W_X_CH1903", "W_Y_CH1903", "W_X", "W_Y"])
    persons = persons.merge(households, on="HHNR", how="left")

    # Negative or zero values are survey codes for "missing", not real coordinates.
    x_lv03 = persons["W_X_CH1903"].where(persons["W_X_CH1903"] > 0)
    y_lv03 = persons["W_Y_CH1903"].where(persons["W_Y_CH1903"] > 0)

    to_lv95 = Transformer.from_crs(CRS_LV03, CRS_LV95, always_xy=True)
    persons["home_x"], persons["home_y"] = to_lv95.transform(x_lv03.values, y_lv03.values)
    return persons


def filter_home_inside(persons, boundary):
    """Step 2b: keep only persons whose home lies inside the study area."""
    n_before = len(persons)
    is_inside = inside(persons["home_x"], persons["home_y"], boundary)
    persons = persons[is_inside].copy()
    print(f"[MZ step 2] persons: {n_before} -> {len(persons)} living inside the study area")
    return persons

def load_trips(persons):
    """MZ step 3: the trips (Wege) of the persons we kept, with origin/destination in LV95."""
    columns = ["HHNR", "WEGNR", "f51100", "f51400", "pseudo",
               "S_X_CH1903", "S_Y_CH1903", "Z_X_CH1903", "Z_Y_CH1903",
               "wmittel", "wzweck1", "wzweck2", "w_rdist"]
    trips = read_mz("wege", columns)
    n_all = len(trips)

    trips = trips[trips["HHNR"].isin(persons["HHNR"])].copy()

    to_lv95 = Transformer.from_crs(CRS_LV03, CRS_LV95, always_xy=True)
    for end, prefix in [("origin", "S"), ("destination", "Z")]:
        x = trips[f"{prefix}_X_CH1903"].where(trips[f"{prefix}_X_CH1903"] > 0)
        y = trips[f"{prefix}_Y_CH1903"].where(trips[f"{prefix}_Y_CH1903"] > 0)
        trips[f"{end}_x"], trips[f"{end}_y"] = to_lv95.transform(x.values, y.values)
        # missing input can come back as 'infinite': turn that into a proper missing value
        trips[[f"{end}_x", f"{end}_y"]] = trips[[f"{end}_x", f"{end}_y"]].replace([np.inf, -np.inf], np.nan)

    trips = trips.sort_values(["HHNR", "WEGNR"])
    print(f"[MZ step 3] trips: {n_all} in total -> {len(trips)} made by the persons we kept")
    return trips

HOME_RADIUS = 100  # metres: a trip starts/ends "at home" if it is this close to the home location


def harmonise_trips(trips, persons, boundary):
    """MZ steps 5-7: common modes and purposes, 'outside' like in MATSim, trips per person."""
    trips = trips.merge(persons[["HHNR", "home_x", "home_y"]], on="HHNR")

    # --- mode -----------------------------------------------------------------
    unmapped = set(trips["wmittel"]) - set(MZ_MODE_MAP)
    assert not unmapped, f"wmittel codes without a mapping: {unmapped}"
    trips["mode"] = trips["wmittel"].map(MZ_MODE_MAP)

    stages = read_mz("etappen", ["HHNR", "WEGNR", "f51300"])
    passenger = stages.loc[stages["f51300"] == MZ_STAGE_CAR_PASSENGER, ["HHNR", "WEGNR"]].drop_duplicates()
    passenger["has_passenger_stage"] = True
    trips = trips.merge(passenger, on=["HHNR", "WEGNR"], how="left")
    trips.loc[(trips["mode"] == "car") & trips["has_passenger_stage"].eq(True), "mode"] = "car_passenger"

    # --- destination purpose: 'going home' (wzweck2 = 2 or 3) wins over wzweck1
    unmapped = set(trips["wzweck1"]) - set(MZ_PURPOSE_MAP)
    assert not unmapped, f"wzweck1 codes without a mapping: {unmapped}"
    trips["destination_purpose_detailed"] = trips["wzweck1"].map(MZ_PURPOSE_MAP)
    trips.loc[trips["wzweck2"] >= 2, "destination_purpose_detailed"] = "home"

    # --- origin purpose: where the previous trip ended; first trip: home if it starts at home
    trips = trips.sort_values(["HHNR", "WEGNR"])
    trips["origin_purpose_detailed"] = trips.groupby("HHNR")["destination_purpose_detailed"].shift(1)
    first_trip = trips["origin_purpose_detailed"].isna()
    starts_at_home = np.hypot(trips["origin_x"] - trips["home_x"],
                              trips["origin_y"] - trips["home_y"]) < HOME_RADIUS
    trips.loc[first_trip & starts_at_home, "origin_purpose_detailed"] = "home"
    trips.loc[first_trip & ~starts_at_home, "origin_purpose_detailed"] = "other"

    # --- outside: places outside the boundary become 'outside', exactly like in MATSim
    for end in ["origin", "destination"]:
        has_coords = trips[f"{end}_x"].notna()
        is_outside = has_coords & ~inside(trips[f"{end}_x"], trips[f"{end}_y"], boundary)
        trips.loc[is_outside, f"{end}_purpose_detailed"] = "outside"

    n_before = len(trips)
    both_outside = ((trips["origin_purpose_detailed"] == "outside")
                    & (trips["destination_purpose_detailed"] == "outside"))
    trips = trips[~both_outside].copy()
    trips["crosses_boundary"] = ((trips["origin_purpose_detailed"] == "outside")
                                 | (trips["destination_purpose_detailed"] == "outside"))
    print(f"[MZ step 5] trips: {n_before} -> {len(trips)} after removing outside-outside trips")

    for end in ["origin", "destination"]:
        trips[f"{end}_purpose"] = fold_purpose(trips[f"{end}_purpose_detailed"])

    # --- same units as MATSim: metres and seconds after midnight
    trips["crowfly_distance"] = np.hypot(trips["destination_x"] - trips["origin_x"],
                                         trips["destination_y"] - trips["origin_y"])
    trips["departure_time"] = trips["f51100"] * 60
    trips["trip_seq"] = trips.groupby("HHNR").cumcount()

    # --- persons with any unknown mode or purpose are dropped (as in eqasim)
    unknown = ((trips["mode"] == "unknown")
               | (trips["origin_purpose"] == "unknown")
               | (trips["destination_purpose"] == "unknown"))
    bad_persons = trips.loc[unknown, "HHNR"].unique()
    persons = persons[~persons["HHNR"].isin(bad_persons)].copy()
    trips = trips[trips["HHNR"].isin(persons["HHNR"])].copy()
    print(f"[MZ step 6] dropped {len(bad_persons)} persons with an unknown mode or purpose")

    # --- step 7: trips per person and stay-at-home, same columns as MATSim
    n_trips = trips.groupby("HHNR").size()
    persons["n_trips"] = persons["HHNR"].map(n_trips).fillna(0).astype(int)
    persons["is_mobile"] = persons["n_trips"] > 0
    persons["weight"] = persons["WP"]
    persons["person_id"] = persons["HHNR"]
    trips["person_id"] = trips["HHNR"]
    return persons, trips

def mz_subscription(ga, ht, va):
    """Combine the three yes/no pass columns into MATSim's categories (1 = yes)."""
    sub = pd.Series("none", index=ga.index)
    sub[ht == 1] = "HT"
    sub[va == 1] = "VA"
    sub[(va == 1) & (ht == 1)] = "VA+HT"
    sub[ga == 1] = "GA"   # a GA includes everything else
    return sub


def add_person_attributes(persons):
    """Age class, car availability, household type, PT pass and PT quality, in the shared categories."""
    target = read_mz("zielpersonen", ["HHNR", "alter", "f42100e", "f41610a", "f41610b", "f41610c"])
    households = read_mz("haushalte", ["HHNR", "hhtyp", "W_OeV_KLASSE"])  # NEW: W_OeV_KLASSE added
    persons = persons.merge(target, on="HHNR", how="left").merge(households, on="HHNR", how="left")

    persons["age"] = persons["alter"]
    persons["age_class"] = age_class(persons["age"])
    persons["car_avail"] = persons["f42100e"].map(CAR_MZ)
    persons["household_type"] = persons["hhtyp"].map(HH_TYPE_MZ)
    persons["subscription"] = mz_subscription(persons["f41610a"], persons["f41610b"], persons["f41610c"])
    persons["pt_quality"] = persons["W_OeV_KLASSE"].map({1: "A", 2: "B", 3: "C", 4: "D", 5: "none"})  # NEW
    return persons