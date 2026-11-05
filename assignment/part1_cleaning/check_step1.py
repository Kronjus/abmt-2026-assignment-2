from microcensus import read_mz, load_persons

all_persons = read_mz("zielpersonen", ["HHNR", "tag", "WP"])
print("Persons per weekday (before filter):")
print(all_persons["tag"].value_counts().sort_index().to_string())
print()

persons = load_persons()
print("Weekdays left:          ", sorted(persons["tag"].unique()))
print("One row per person:     ", persons["HHNR"].is_unique)
print("Missing weights:        ", persons["WP"].isna().sum())
print("Share of all persons:   ", round(len(persons) / len(all_persons), 3))