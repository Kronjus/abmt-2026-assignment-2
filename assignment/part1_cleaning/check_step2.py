import numpy as np
from pyproj import Transformer

from config import load_boundary, CRS_WGS84, CRS_LV95
from microcensus import load_persons, add_home_coordinates, filter_home_inside

persons = load_persons()
persons = add_home_coordinates(persons)
print("Persons without valid home coordinates:", int(persons["home_x"].isna().sum()))

# Cross-check: convert the longitude/latitude columns too and compare
from_wgs84 = Transformer.from_crs(CRS_WGS84, CRS_LV95, always_xy=True)
x2, y2 = from_wgs84.transform(persons["W_X"].values, persons["W_Y"].values)
diff = np.hypot(persons["home_x"] - x2, persons["home_y"] - y2)
print("LV03 vs WGS84 difference (m): median", round(float(np.nanmedian(diff)), 1),
      "| max", round(float(np.nanmax(diff)), 1))

boundary = load_boundary()
persons_inside = filter_home_inside(persons, boundary)
print("Share of weekday persons inside:", round(len(persons_inside) / len(persons), 3))