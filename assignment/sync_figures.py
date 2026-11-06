"""Copy the figures and tables that the report uses into data/, the folder that is uploaded to Overleaf.

Usage (from the repository root, after the analysis scripts have run):
    .venv/Scripts/python assignment/sync_figures.py

Every `\\includegraphics{figures/<name>}` and every `\\input{tables/<name>}` in data/*.tex is looked up by file name in
the output folders of the analysis scripts (assignment/*/output/) and copied to data/figures/<name> or
data/tables/<name>. Names that are not found or that exist in several output folders are reported. The data/ folder
is git-ignored, so the copies are not under version control.
"""
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
KINDS = {"figures": re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{figures/([^}]+)\}"),
         "tables": re.compile(r"\\input\{tables/([^}]+)\}")}


def main():
    total = copied = 0
    for folder, pattern in KINDS.items():
        names = sorted({m for tex in DATA.glob("*.tex") for m in pattern.findall(tex.read_text(encoding="utf-8"))})
        (DATA / folder).mkdir(exist_ok=True)
        for name in names:
            total += 1
            found = list((ROOT / "assignment").glob(f"*/output/{name}"))
            if len(found) != 1:
                print(f"NOT COPIED: {folder}/{name} ({'not found' if not found else f'{len(found)} candidates'})")
                continue
            shutil.copy2(found[0], DATA / folder / name)
            copied += 1
            print(f"{found[0].relative_to(ROOT)} -> {(DATA / folder / name).relative_to(ROOT)}")
    print(f"{copied} of {total} files copied")


if __name__ == "__main__":
    main()
