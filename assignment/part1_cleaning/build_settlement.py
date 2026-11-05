"""Build the settlement-type map: BFS municipality boundaries (G1, 01.01.2023, LV95)
+ Stadt/Land-Typologie 2012 from Raumgliederungen.xlsx (same reference date)."""
import geopandas as gpd
import pandas as pd

from config import DATA, CRS_LV95, load_boundary

FOLDER = DATA / "stadtland_typologie"
OUTPUT = FOLDER / "settlement_2023.gpkg"
STALAN = {1: "urban", 2: "intermediate", 3: "rural"}

# --- 1. municipality boundaries --------------------------------------------
# BFS naming: g1 = generalisation level 1, g = Gemeinden (municipalities), 23 = 2023
print("Shapefiles found:", sorted(f.name for f in FOLDER.rglob("*.shp")))
candidates = sorted(FOLDER.rglob("g1g23*.shp"))
assert candidates, "No municipality file g1g23*.shp found - paste the list above"
munis = gpd.read_file(candidates[0])
print("Using:", candidates[0].name)
print("Columns:", list(munis.columns), "| CRS:", munis.crs, "| municipalities:", len(munis))

munis = munis.to_crs(CRS_LV95) if munis.crs else munis.set_crs(CRS_LV95)

# --- 2. typology table -----------------------------------------------------
# Sheet 'Daten': 3 header rows, column 0 = BFS municipality number, column 6 = Stadt/Land-Typologie
table = pd.read_excel(FOLDER / "Raumgliederungen.xlsx", sheet_name="Daten", header=None,
                      skiprows=3, usecols=[0, 6], names=["bfs_nr", "stalan"]).dropna().astype(int)
print("Typology table:", len(table), "municipalities")

# --- 3. join via the municipality number -----------------------------------
nr_col = next(c for c in ["GMDNR", "BFS_NUMMER", "GDENR", "BFSNR"] if c in munis.columns)
munis = munis.merge(table, left_on=nr_col, right_on="bfs_nr", how="left")
munis["settlement"] = munis["stalan"].map(STALAN)
print(f"Joined on '{nr_col}': {int(munis['settlement'].isna().sum())} municipalities without a class")

# --- 4. quick look at the study area and save -------------------------------
in_area = munis[munis.intersects(load_boundary())]
print("Municipalities touching the study area, by class:", in_area["settlement"].value_counts().to_dict())

munis[["bfs_nr", "settlement", "geometry"]].to_file(OUTPUT, driver="GPKG")
print("Saved", OUTPUT)