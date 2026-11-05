from matsim import load_population, is_stage_activity

persons, activities, trips = load_population()

print("Persons:", len(persons), "| activities:", len(activities), "| trips:", len(trips))
print()
print("Freight agents:     ", int(persons["isFreight"].fillna(False).sum()))
print("Cross-border agents:", int(persons["isCrossBorder"].fillna(False).sum()))
if "subpopulation" in persons.columns:
    print("Subpopulation values:", persons["subpopulation"].value_counts(dropna=False).to_dict())
print()
print("Stage activities left:", int(activities["type"].map(is_stage_activity).sum()))
print("Activity types:", activities["type"].value_counts().to_dict())
print("Trip modes:    ", trips["mode"].value_counts(dropna=False).to_dict())
print()
trips_per_person = trips.groupby("person_id").size().reindex(persons["person_id"], fill_value=0)
print("Persons with 0 trips:", int((trips_per_person == 0).sum()))
print()
print("Example person 10005229 (the one we looked at):")
cols = ["trip_seq", "origin_type", "destination_type", "mode", "departure_time", "crowfly_distance"]
print(trips[trips["person_id"] == "10005229"][cols].to_string(index=False))