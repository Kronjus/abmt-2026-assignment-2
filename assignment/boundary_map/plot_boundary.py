"""Draw the Fribourg filtering boundary on an interactive map.

Usage (from the repository root):
    .venv/Scripts/python assignment/boundary_map/plot_boundary.py [path/to/boundary.shp]

Writes assignment/boundary_map/output/boundary_map.html.
"""
import sys
from pathlib import Path

import folium
import geopandas as gpd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHAPEFILE = ROOT / "data" / "Boundary" / "fribourg_convexhull.shp"
OUTPUT = Path(__file__).resolve().parent / "output" / "boundary_map.html"

# The .prj is an ESRI-style CH1903+ / LV95 definition that pyproj may not map
# to an EPSG code, so set it explicitly.
SOURCE_CRS = "EPSG:2056"

shapefile = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SHAPEFILE
boundary = gpd.read_file(shapefile)
boundary = boundary.set_crs(SOURCE_CRS, allow_override=True)

area_km2 = boundary.geometry.area.sum() / 1e6
print(f"{shapefile.name}: {len(boundary)} feature(s), {area_km2:.1f} km2, bounds (LV95) {boundary.total_bounds}")

# Attributes (e.g. timestamps) are not needed for drawing and break JSON export.
boundary = boundary[["geometry"]].to_crs("EPSG:4326")
minx, miny, maxx, maxy = boundary.total_bounds

fmap = folium.Map(tiles="OpenStreetMap")
folium.GeoJson(
    boundary,
    name="Boundary",
    style_function=lambda _: {"color": "#d62728", "weight": 3, "fillColor": "#d62728", "fillOpacity": 0.08},
).add_to(fmap)
fmap.fit_bounds([[miny, minx], [maxy, maxx]])

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fmap.save(OUTPUT)
print(f"Wrote {OUTPUT}")
