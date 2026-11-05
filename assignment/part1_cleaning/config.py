"""Shared settings for Part 1: file paths, coordinate systems and the boundary filter."""
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely

# --- Paths -------------------------------------------------------------------
# ROOT is the project folder: this file lives in assignment/part1_cleaning/,
# so we go two folders up.
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

MZ_DIR = DATA / "microcensus"
MATSIM_POPULATION = DATA / "Fribourg_10pct" / "fribourg_population.xml.gz"
BOUNDARY_FILE = DATA / "Fribourg_10pct" / "Boundary" / "fribourg_convexhull.shp"
CLEAN_DIR = DATA / "clean"

# --- Coordinate systems ------------------------------------------------------
CRS_LV03 = "EPSG:21781"  # old Swiss system, used by the Mikrozensus *_CH1903 columns
CRS_LV95 = "EPSG:2056"   # current Swiss system, used by MATSim and the boundary
CRS_WGS84 = "EPSG:4326"  # longitude / latitude


def load_boundary():
    """Read the study-area polygon and return it as one shapely geometry in LV95."""
    gdf = gpd.read_file(BOUNDARY_FILE)
    # The .prj file is not recognised, so we state the coordinate system ourselves.
    gdf = gdf.set_crs(CRS_LV95, allow_override=True)
    boundary = shapely.union_all(gdf.geometry.values)  # merge all parts into one shape
    shapely.prepare(boundary)  # makes the many point-in-polygon tests much faster
    return boundary


def inside(x, y, boundary):
    """For arrays of x and y (LV95), return True where the point lies in the boundary.
    Missing coordinates (NaN) count as outside."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    result = np.zeros(len(x), dtype=bool)
    valid = ~(np.isnan(x) | np.isnan(y))
    points = shapely.points(x[valid], y[valid])
    result[valid] = shapely.covers(boundary, points)
    return result