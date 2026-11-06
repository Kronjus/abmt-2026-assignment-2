import pandas as pd

from config import CLEAN_DIR

persons = pd.read_csv(CLEAN_DIR / "microcensus" / "persons.csv")
trips = pd.read_csv(CLEAN_DIR / "microcensus" / "trips.csv").merge(
    persons[["person_id", "weight"]], on="person_id")

def mode_shares(t):
    s = t.groupby("mode")["weight"].sum()
    return (100 * s / s.sum()).round(1)

crossing = trips[trips["crosses_boundary"]]
inside_only = trips[~trips["crosses_boundary"]]

print(f"Share of trips crossing the boundary (weighted): "
      f"{100 * crossing['weight'].sum() / trips['weight'].sum():.1f}%")
print("\nMode shares (%):")
print(pd.DataFrame({"all trips": mode_shares(trips),
                    "crossing only": mode_shares(crossing),
                    "inside only": mode_shares(inside_only)}).to_string())
print("\nMedian crow-fly distance (km): crossing",
      round(crossing["crowfly_distance"].median() / 1000, 1),
      "| inside only", round(inside_only["crowfly_distance"].median() / 1000, 1))