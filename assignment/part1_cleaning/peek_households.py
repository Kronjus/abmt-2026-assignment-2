"""Print the first three households (synthetic data, fine to print)."""
import gzip
import xml.etree.ElementTree as ET

from config import DATA

HOUSEHOLDS = DATA / "Fribourg_10pct" / "fribourg_households.xml.gz"

count = 0
with gzip.open(HOUSEHOLDS, "rb") as f:
    for _, elem in ET.iterparse(f, events=("end",)):
        if elem.tag.endswith("household"):
            ET.indent(elem)
            print(ET.tostring(elem, encoding="unicode"))
            count += 1
            if count == 3:
                break