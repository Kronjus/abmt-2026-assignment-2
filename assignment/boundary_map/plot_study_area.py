"""Static map of the study area for the report: the filtering boundary over the swisstopo basemap, with the
municipalities coloured by the BFS Stadt/Land-Typologie 2012 (urban, intermediate, rural).

Usage (from the repository root):
    .venv/Scripts/python assignment/boundary_map/plot_study_area.py

Writes assignment/boundary_map/output/study_area.png. Only the boundary and municipalities are drawn, no persons.
The basemap tiles (free geo.admin.ch WMTS) are downloaded when the script runs.
"""
from pathlib import Path

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
BOUNDARY = ROOT / "data" / "Boundary" / "fribourg_convexhull.shp"
STADTLAND = ROOT / "data" / "stadtland_typologie" / "Boundaries_G1_GDETYP2012_20230101_de.gpkg"
OUT = Path(__file__).resolve().parent / "output" / "study_area.png"
BASEMAP = "https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-grau/default/current/3857/{z}/{x}/{y}.jpeg"
CLASSES = {"1": ("urban", "#d95f02"), "2": ("intermediate", "#e6ab02"), "3": ("rural", "#1b9e77")}

hull = gpd.read_file(BOUNDARY).set_crs("EPSG:2056", allow_override=True)
zones = gpd.read_file(STADTLAND, columns=["STALAN2012"]).to_crs("EPSG:2056")
zones = gpd.clip(zones, hull.geometry.union_all())  # only the municipalities inside the study area (cut at the boundary)

fig, ax = plt.subplots(figsize=(6, 6.4))
minx, miny, maxx, maxy = hull.total_bounds
pad = 0.03 * max(maxx - minx, maxy - miny)
ax.set_xlim(minx - pad, maxx + pad)
ax.set_ylim(miny - pad, maxy + pad)
for key, (_, color) in CLASSES.items():
    zones[zones["STALAN2012"] == key].plot(ax=ax, color=color, alpha=0.45, edgecolor="white", linewidth=0.3, zorder=2)
cx.add_basemap(ax, crs="EPSG:2056", source=BASEMAP, zoom=11, zorder=1, attribution=False)
hull.boundary.plot(ax=ax, color="black", linewidth=1.4, zorder=4)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("")
ax.set_ylabel("")
ax.legend(handles=[Patch(facecolor=c, alpha=0.6, label=n) for n, c in CLASSES.values()] +
                  [Patch(facecolor="none", edgecolor="black", label="study area")],
          loc="upper left", frameon=True, fontsize=8, title="Settlement pattern", title_fontsize=8)
fig.text(0.99, 0.01, "Basemap: © swisstopo (Landeskarte grau); BFS Stadt/Land-Typologie 2012", ha="right", fontsize=6)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight", pad_inches=0.05)
print(f"Wrote {OUT}")
