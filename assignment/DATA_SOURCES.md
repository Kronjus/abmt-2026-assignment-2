# Data sources

The `data/` folder is git-ignored, so this file records what is in it and where it came from.
Place the files at the paths below to reproduce the assignment scripts.

Entries marked **TODO** were not verified; fill them in when known.

## External open data

### ÖV-Güteklassen (public transport quality classes)

| | |
|---|---|
| Path | `data/gueteklassen_oev_2026/OeV_Gueteklassen_ARE.gpkg` (layers `OeV_Gueteklassen_ARE`, polygons; `OeV_Haltestellen_ARE`, points) |
| Also in folder | `Oev_Haltestellen_ARE.xlsx` (stops) |
| Publisher | Federal Office for Spatial Development (ARE) |
| Catalogue | [opendata.swiss: ÖV-Güteklassen ARE](https://opendata.swiss/en/dataset/ov-guteklassen-are1) (federal data on geo.admin.ch, collection `ch.are.gueteklassen_oev`) |
| CRS | EPSG:2056 (LV95) |
| Content | Polygons with `KLASSE` = A (very good) to D; locations outside every polygon have no class |
| License | Open use, source must be credited (per the opendata.swiss entry) |
| Version / download date | 2026 edition per the folder name; exact download date **TODO** |
| Used in | `assignment/compare_distributions/compare.py`: PT quality of each population home (point-in-polygon; best class if polygons overlap; none if outside) |

### Stadt/Land-Typologie 2012 (settlement pattern)

| | |
|---|---|
| Path | `data/stadtland_typologie/Boundaries_G1_GDETYP2012_20230101_de.gpkg` (layer `boundaries`) |
| Publisher | Federal Statistical Office (BFS), "Raumgliederungen der Schweiz" (municipality boundaries as of 2023-01-01, per the file name) |
| Download URL | [bfs.admin.ch: Raumgliederungen asset 24126179](https://www.bfs.admin.ch/bfs/rm/home.assetdetail.24126179.html) |
| CRS | EPSG:2056 (LV95) |
| Content | 2139 municipalities. `STALAN2012` (stored as text): 1 = urban, 2 = intermediate, 3 = rural. Also `GDETYP2012_9` and `GDETYP2012_25` (finer typologies), unused. Some polygons have no class |
| License | OPEN BY (open use, attribution required) |
| Used in | `assignment/compare_distributions/compare.py`: settlement pattern of population homes and microcensus homes. Points on unclassified polygons take the nearest classified municipality |

## Project data (not open data)

| Path | What | Source | Used in |
|---|---|---|---|
| `data/Fribourg_10pct/` | Synthetic population, households, network, schedule, config for Fribourg (10 % sample) | **TODO** (course material) | `filter_data`, `compare_distributions` |
| `data/microcensus/` | Swiss microcensus on mobility and transport (MZMV 2015), CSV tables and codebook `2017-05-08-Datenbankstruktur_MZMV2015_mit_Geodaten.xlsx` | **TODO** (course material, access restricted) | `filter_data`, `compare_distributions` |
| `data/Boundary/fribourg_convexhull.*` | Study-area boundary (shapefile, EPSG:2056) | **TODO** | `filter_data`, `boundary_map` |
| `data/filtered/` | Generated: population and microcensus filtered to the boundary and to weekday reports | Output of `assignment/filter_data/filter_data.py` | `compare_distributions` |

## Adding a source

Add a table with path, publisher, URL, CRS, license, download date and the script that uses it.
