"""Read the xlsx files of analysis/data/raw/publication_data with the standard library only.

Cells come back as strings exactly as stored in the sheet XML (numbers as their
stored decimal text, e.g. "0.9996999999999945"); empty cells are None.
"""

from __future__ import annotations

import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
RID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"



def utf8_stdout() -> None:
    """The Windows console is cp1252; sheet names and sites carry umlauts."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def _col(ref: str) -> int:
    n = 0
    for ch in re.match(r"[A-Z]+", ref).group(0):
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def read_xlsx(path: Path) -> dict[str, list[tuple]]:
    """Return {sheet name: rows} in workbook order; row 0 is the header row."""
    z = zipfile.ZipFile(path)
    shared: list[str] = []
    if "xl/sharedStrings.xml" in z.namelist():
        root = ET.fromstring(z.read("xl/sharedStrings.xml"))
        for si in root.findall("m:si", NS):
            shared.append("".join(t.text or "" for t in si.iter("{%s}t" % NS["m"])))
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    relmap = {r.get("Id"): r.get("Target") for r in rels}
    sheets: dict[str, list[tuple]] = {}
    for s in wb.find("m:sheets", NS):
        target = relmap[s.get(RID)].lstrip("/")
        if not target.startswith("xl/"):
            target = "xl/" + target
        root = ET.fromstring(z.read(target))
        rows = []
        for row in root.iter("{%s}row" % NS["m"]):
            vals = {}
            for c in row.findall("m:c", NS):
                t = c.get("t")
                v = c.find("m:v", NS)
                if t == "s" and v is not None:
                    val = shared[int(v.text)]
                elif t == "inlineStr":
                    val = "".join(x.text or "" for x in c.iter("{%s}t" % NS["m"]))
                else:
                    val = v.text if v is not None else None
                vals[_col(c.get("r"))] = val
            if vals:
                rows.append(tuple(vals.get(i) for i in range(max(vals) + 1)))
        sheets[s.get("name")] = rows
    return sheets


def records(rows: list[tuple]) -> list[dict]:
    """Rows as dicts keyed by the header row (missing trailing cells -> None)."""
    head = rows[0]
    return [{h: (r[i] if i < len(r) else None) for i, h in enumerate(head)} for r in rows[1:]]


def num(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None
