"""Map the use of the bus lines in the simulation and highlight the underused ones.

Usage (from the repository root):
    .venv/Scripts/python assignment/compare_distributions/bus_routes.py

Outputs (assignment/compare_distributions/output/):
    bus_routes.csv            one row per bus line with departures, passengers and load factor
    bus_routes_usage.png      map of all bus links, width and colour = passengers per day
    bus_routes_underused.png  map of the underused lines (red) next to a ranking of the lowest load factors

Notes
- Data: output_transitSchedule, output_transitVehicles, output_network and pt_link_volumes.csv.gz of
  simulation_output/. A bus line is a transit line whose routes have transport mode `bus`; all its routes
  (variants and both directions) are pooled.
- The simulation is a 10 % sample of the persons, but the timetable is complete, so passenger numbers are
  scaled by 1 / SAMPLE_SIZE to full scale before they are compared with the vehicle capacity.
- Load factor of a line = passenger-km / seat-km. Passenger-km: pt_link_volumes (passengers per link, route and
  time bin) times the link length. Seat-km: for every departure, the capacity of its vehicle type (seats plus
  standing room) times the length of the route. A load factor of 10 % means the vehicles run 90 % empty.
- A line is "underused" if its load factor is below UNDERUSED and it has at least MIN_DEPARTURES departures per
  day. Both thresholds are choices, not rules. Lines with fewer departures are drawn in light grey and not rated.
- Bus lines are drawn along their network links, so the lines follow the roads of the simulation.
"""
import gzip
import xml.etree.ElementTree as ET
from collections import defaultdict

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from activity_map import BASEMAP
from mode_share import BOUNDARY, OUT, SIM

SAMPLE_SIZE = 0.1
UNDERUSED = 0.10  # load factor below which a line counts as underused
MIN_DEPARTURES = 10  # lines with fewer departures per day (night and special lines) are not rated: with a 10 % sample, a single passenger is 10 passengers at full scale
N_RANKED = 15


def read_network():
    nodes, links = {}, {}
    for _, el in ET.iterparse(gzip.open(SIM / "output_network.xml.gz"), events=("end",)):
        if el.tag == "node":
            nodes[el.get("id")] = (float(el.get("x")), float(el.get("y")))
        elif el.tag == "link":
            links[el.get("id")] = (el.get("from"), el.get("to"), float(el.get("length")))
        if el.tag in ("node", "link"):
            el.clear()
    return nodes, links


def read_capacity():
    capacity, vehicle_type = {}, {}
    for _, el in ET.iterparse(gzip.open(SIM / "output_transitVehicles.xml.gz"), events=("end",)):
        tag = el.tag.split("}")[-1]
        if tag == "vehicleType":
            cap = el.find("{*}capacity")
            capacity[el.get("id")] = int(cap.get("seats")) + int(cap.get("standingRoomInPersons"))
        elif tag == "vehicle":
            vehicle_type[el.get("id")] = el.get("type")
    return capacity, vehicle_type


def read_bus_routes(links, capacity, vehicle_type):
    """Per bus route: line id and name, route links, number of departures and seats offered per route-km."""
    routes = []
    for _, el in ET.iterparse(gzip.open(SIM / "output_transitSchedule.xml.gz"), events=("end",)):
        if el.tag != "transitLine":
            continue
        for r in el.findall("transitRoute"):
            if r.findtext("transportMode") != "bus":
                continue
            route_links = [l.get("refId") for l in r.findall("route/link")]
            length_km = sum(links[l][2] for l in route_links if l in links) / 1000
            seats = [capacity[vehicle_type[d.get("vehicleRefId")]] for d in r.findall("departures/departure")]
            routes.append({"line_id": el.get("id"), "line_name": el.get("name"), "route_id": r.get("id"),
                           "links": route_links, "length_km": length_km, "departures": len(seats),
                           "seat_km": length_km * sum(seats)})
        el.clear()
    return pd.DataFrame(routes)


def line_table(routes, volumes, links):
    volumes = volumes.assign(length_km=volumes["linkId"].astype(str).map(lambda l: links[l][2] / 1000 if l in links else np.nan))
    volumes = volumes[volumes["routeId"].isin(routes["route_id"])]
    pax_km = (volumes["passengers"] * volumes["length_km"] / SAMPLE_SIZE).groupby(volumes["lineId"]).sum()
    lines = routes.groupby("line_id").agg(line_name=("line_name", "first"), routes=("route_id", "size"),
                                          departures=("departures", "sum"), seat_km=("seat_km", "sum"))
    lines["passenger_km"] = pax_km.reindex(lines.index).fillna(0.0)
    lines["load_factor"] = lines["passenger_km"] / lines["seat_km"]
    return lines[lines["seat_km"] > 0]


def segments(link_ids, nodes, links):
    out = []
    for l in link_ids:
        if l in links and links[l][0] in nodes and links[l][1] in nodes:
            out.append((nodes[links[l][0]], nodes[links[l][1]]))
    return out


def base_map(ax, hull):
    minx, miny, maxx, maxy = hull.bounds
    pad = 0.04 * max(maxx - minx, maxy - miny)
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    cx.add_basemap(ax, crs="EPSG:2056", source=BASEMAP, zoom=12, zorder=1, attribution=False)
    gpd.GeoSeries([hull], crs="EPSG:2056").boundary.plot(ax=ax, color="black", linewidth=0.8, zorder=4)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")


def plot_usage(volumes, nodes, links, hull):
    per_link = (volumes.groupby(volumes["linkId"].astype(str))["passengers"].sum() / SAMPLE_SIZE).sort_values()
    per_link = per_link[per_link.index.isin(links)]
    segs = segments(per_link.index, nodes, links)
    values = per_link.to_numpy()
    fig, ax = plt.subplots(figsize=(8, 8))
    base_map(ax, hull)
    norm = plt.Normalize(0, np.percentile(values, 98))
    lc = LineCollection(segs, array=values, cmap="plasma", norm=norm,
                        linewidths=0.6 + 3.2 * np.sqrt(np.clip(values / values.max(), 0, 1)), zorder=3)
    ax.add_collection(lc)
    fig.colorbar(lc, ax=ax, shrink=0.7, label="bus passengers per day and link (full scale)")
    ax.set_title("Bus use in the simulation")
    fig.text(0.99, 0.01, "Basemap: © swisstopo (Landeskarte grau)", ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "bus_routes_usage.png", dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def plot_underused(lines, routes, nodes, links, hull):
    rated = lines[lines["departures"] >= MIN_DEPARTURES]
    low = rated[rated["load_factor"] < UNDERUSED]
    is_low = routes["line_id"].isin(low.index)
    is_rare = routes["line_id"].isin(lines.index[lines["departures"] < MIN_DEPARTURES])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1.15, 1]})
    base_map(ax, hull)
    for mask, color, width, z in ((is_rare, "#c9ccd1", 0.8, 2), (~is_low & ~is_rare, "#6b7280", 0.8, 3), (is_low, "#d7191c", 1.8, 4)):
        links_used = {l for ids in routes.loc[mask, "links"] for l in ids}
        ax.add_collection(LineCollection(segments(links_used, nodes, links), colors=color, linewidths=width, zorder=z))
    ax.plot([], [], color="#6b7280", label=f"other lines ({len(rated) - len(low)})")
    ax.plot([], [], color="#c9ccd1", label=f"fewer than {MIN_DEPARTURES} departures, not rated ({len(lines) - len(rated)})")
    ax.plot([], [], color="#d7191c", label=f"underused: load factor < {UNDERUSED:.0%} ({len(low)})")
    ax.legend(loc="lower left", frameon=True, fontsize=8)
    ax.set_title("Underused bus lines")

    ranked = rated.sort_values("load_factor").head(N_RANKED).iloc[::-1]
    labels = [f"{n} ({int(d)} dep.)" for n, d in zip(ranked["line_name"], ranked["departures"])]
    ax2.barh(labels, 100 * ranked["load_factor"], color=["#d7191c" if v < UNDERUSED else "#6b7280" for v in ranked["load_factor"]])
    overall = 100 * lines["passenger_km"].sum() / lines["seat_km"].sum()
    ax2.axvline(overall, color="black", linestyle=":")
    ax2.text(overall, 0.01, f" all lines: {overall:.0f} %", fontsize=8, va="bottom", ha="left", transform=ax2.get_xaxis_transform())
    ax2.set_xlabel("load factor [% of seat-km occupied]")
    ax2.set_title(f"The {N_RANKED} rated lines with the lowest load factor")
    fig.text(0.99, 0.01, "Basemap: © swisstopo (Landeskarte grau)", ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "bus_routes_underused.png", dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main():
    nodes, links = read_network()
    capacity, vehicle_type = read_capacity()
    routes = read_bus_routes(links, capacity, vehicle_type)
    volumes = pd.read_csv(SIM / "pt_link_volumes.csv.gz", sep=";", dtype={"linkId": str})
    volumes = volumes[volumes["routeId"].isin(routes["route_id"])]
    lines = line_table(routes, volumes, links)
    hull = gpd.read_file(BOUNDARY).set_crs("EPSG:2056", allow_override=True).geometry.union_all()

    print(f"{len(lines)} bus lines, {len(routes)} routes, {int(lines['departures'].sum())} departures per day")
    print(f"all lines: load factor {lines['passenger_km'].sum() / lines['seat_km'].sum():.1%}")
    print(lines["load_factor"].describe(percentiles=[.1, .25, .5, .75, .9]).round(3).to_string())
    rated = lines[lines["departures"] >= MIN_DEPARTURES]
    low = rated[rated["load_factor"] < UNDERUSED]
    print(f"{len(rated)} lines with at least {MIN_DEPARTURES} departures; underused (< {UNDERUSED:.0%}): {len(low)} lines, "
          f"{low['departures'].sum() / lines['departures'].sum():.0%} of all departures")
    OUT.mkdir(parents=True, exist_ok=True)
    lines.sort_values("load_factor").to_csv(OUT / "bus_routes.csv", float_format="%.3f")
    plot_usage(volumes, nodes, links, hull)
    plot_underused(lines, routes, nodes, links, hull)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
