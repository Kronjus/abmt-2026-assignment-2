"""Print the XML of one synthetic resident, to see the file structure."""
import gzip
import xml.etree.ElementTree as ET

from config import MATSIM_POPULATION

with gzip.open(MATSIM_POPULATION, "rb") as f:
    for event, elem in ET.iterparse(f, events=("end",)):
        if elem.tag != "person":
            continue

        attributes = {a.get("name"): a.text for a in elem.findall("./attributes/attribute")}
        plan = elem.find("plan")
        n_activities = len(plan.findall("activity")) if plan is not None else 0

        if attributes.get("subpopulation") not in ("freight", "crossborder") and n_activities >= 4:
            ET.indent(elem)
            print(ET.tostring(elem, encoding="unicode"))
            break

        elem.clear()  # free memory for persons we don't need