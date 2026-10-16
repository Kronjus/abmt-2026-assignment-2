"""Keep only the population and microcensus records whose home lies inside the boundary.

Usage (from the repository root):
    .venv/Scripts/python assignment/filter_data/filter_data.py

Inputs : data/Boundary/fribourg_convexhull.shp (LV95)
         data/Fribourg_10pct/fribourg_population.xml.gz (home_x/home_y in LV95)
         data/microcensus/*.csv (household home W_X/W_Y in WGS84, keyed by HHNR)
Outputs: data/filtered/fribourg_population.xml.gz
         data/filtered/microcensus/*.csv (same files, only households living inside
         whose reported day (zielpersonen.csv "tag", 1=Mon .. 7=Sun) is Monday-Friday)

The population has no weekday (MATSim simulates one typical day), so only the
microcensus is restricted to weekdays.
"""
import csv
import gzip
import re
import sys
from pathlib import Path

import geopandas as gpd
from pyproj import Transformer
from shapely import prepared, points

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
BOUNDARY = DATA / "Boundary" / "fribourg_convexhull.shp"
POPULATION = DATA / "Fribourg_10pct" / "fribourg_population.xml.gz"
MICROCENSUS = DATA / "microcensus"
OUT = DATA / "filtered"

csv.field_size_limit(sys.maxsize)

polygon = gpd.read_file(BOUNDARY).set_crs("EPSG:2056", allow_override=True).geometry.union_all()
inside = prepared.prep(polygon)


def filter_population():
    home_x = re.compile(r'name="home_x"[^>]*>([^<]+)<')
    home_y = re.compile(r'name="home_y"[^>]*>([^<]+)<')
    kept = total = 0
    block = None
    out_path = OUT / POPULATION.name
    with gzip.open(POPULATION, "rt", encoding="utf-8") as src, gzip.open(out_path, "wt", encoding="utf-8") as dst:
        for line in src:
            if block is None:
                if "<person " in line:
                    block = [line]
                else:
                    dst.write(line)
                continue
            block.append(line)
            if "</person>" in line:
                total += 1
                text = "".join(block)
                x, y = home_x.search(text), home_y.search(text)
                if x and y and inside.covers(points(float(x.group(1)), float(y.group(1)))):
                    dst.write(text)
                    kept += 1
                block = None
    print(f"population: kept {kept} of {total} persons -> {out_path}")


def read_csv_rows(path):
    with open(path, newline="", encoding="latin1") as f:
        yield from csv.reader(f)


def household_ids():
    rows = read_csv_rows(MICROCENSUS / "haushalte.csv")
    header = next(rows)
    i_hh, i_x, i_y = header.index("HHNR"), header.index("W_X"), header.index("W_Y")
    ids, xs, ys = [], [], []
    for row in rows:
        ids.append(row[i_hh])
        xs.append(float(row[i_x]))
        ys.append(float(row[i_y]))
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True).transform(xs, ys)
    keep = {hh for hh, px, py in zip(ids, x, y) if inside.covers(points(px, py))}
    print(f"microcensus: {len(keep)} of {len(ids)} households live inside")
    return keep


def weekday_household_ids():
    rows = read_csv_rows(MICROCENSUS / "zielpersonen.csv")
    header = next(rows)
    i_hh, i_tag = header.index("HHNR"), header.index("tag")
    keep = {row[i_hh] for row in rows if row[i_tag] in {"1", "2", "3", "4", "5"}}
    print(f"microcensus: {len(keep)} households have a Monday-Friday reported day")
    return keep


def filter_microcensus():
    keep = household_ids() & weekday_household_ids()
    print(f"microcensus: {len(keep)} households kept after both filters")
    out_dir = OUT / "microcensus"
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(MICROCENSUS.glob("*.csv")):
        rows = read_csv_rows(path)
        header = next(rows)
        i_hh = header.index("HHNR")
        n_in = n_out = 0
        with open(out_dir / path.name, "w", newline="", encoding="latin1") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(header)
            for row in rows:
                n_in += 1
                if row[i_hh] in keep:
                    writer.writerow(row)
                    n_out += 1
        print(f"  {path.name}: kept {n_out} of {n_in} rows")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    filter_population()
    filter_microcensus()
