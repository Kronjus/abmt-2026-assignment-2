import gzip
import xml.etree.ElementTree as ET

from config import DATA, load_boundary
from matsim import load_population, filter_residents

HOUSEHOLDS = DATA / "Fribourg_10pct" / "fribourg_households.xml.gz"

persons, activities, trips = load_population()
persons, _, _ = filter_residents(persons, activities, trips, load_boundary())

member_of = {}
with gzip.open(HOUSEHOLDS, "rb") as f:
    for _, elem in ET.iterparse(f, events=("end",)):
        if elem.tag.endswith("household"):
            for child in elem.iter():
                if child.tag.endswith("personId"):
                    member_of[child.get("refId")] = elem.get("id")
            elem.clear()

persons["hh"] = persons["person_id"].map(member_of)
members_present = persons.groupby("hh")["person_id"].transform("size")

print("Residents without a household:", int(persons["hh"].isna().sum()))
print("How many members of each resident's household are in the file (% of residents):")
print((members_present.value_counts(normalize=True).sort_index() * 100).round(1).to_string())