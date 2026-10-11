"""Table S14: orders and families of the 53 samples of the main analysis (sample 180 excluded).

Counts the field specimens of specimen_data.xlsx (sheet 'Object List') per order and family
with the same filters as Data_Analysis.Rmd: valid samples only (is_valid == "True"), specimens
without an order dropped, specimens without a family counted at order level, sample 180
(zuordnung_id 180) excluded as in the main analysis. A pure count, no statistics; standard
library only.

The total must equal the specimen count of the main analysis (37,612 specimens in 53 samples,
key specimens_with_order_n in key_numbers.csv of the main run).

Usage (from the repository root):
    python analysis/reproduction/make_table_s14.py [--check]
Writes analysis/reproduction/tables/table_s14_families.yaml; --check compares the generated
table with that file instead of writing it (exit 1 on any difference).
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from xlsx_read import read_xlsx, records, utf8_stdout  # noqa: E402

PD = HERE.parent / "data" / "raw" / "publication_data"
OUT = HERE / "tables" / "table_s14_families.yaml"
EXCLUDED_SAMPLE = "180"
EXPECTED_TOTAL = 37612  # specimens with order in the 53 valid samples of the main analysis
EXPECTED_SAMPLES = 53
NO_FAMILY = "no family assigned"
def fmt(n: int) -> str:
    return f"{n:,}"


def count() -> tuple[collections.Counter, int, int]:
    objs = records(read_xlsx(PD / "specimen_data.xlsx")["Object List"])
    kept = [o for o in objs
            if o["is_valid"] == "True" and o["zuordnung_id"] != EXCLUDED_SAMPLE and o["order"]]
    samples = {o["zuordnung_id"] for o in kept}
    counts = collections.Counter((o["order"], o["family"] or NO_FAMILY) for o in kept)
    return counts, len(kept), len(samples)


def build(counts: collections.Counter, total: int) -> str:
    orders = collections.Counter()
    for (order, _), n in counts.items():
        orders[order] += n
    lines = [
        'title: "Orders and families recorded in the 53 sampling units (excluding sample 180), with '
        'the number of specimens assigned to each by the pipeline. Specimens without a family-level '
        'assignment are listed as \\"no family assigned\\" and were analysed at order level; '
        'specimens without an order-level assignment were excluded from the analyses and are not '
        'listed."',
        "columns:",
        '  - name: "Order"',
        "    width: 2.4",
        "    align: left",
        '  - name: "Family"',
        "    width: 2.8",
        "    align: left",
        '  - name: "Specimens"',
        "    width: 1.3",
        "    align: center",
        "data:",
    ]
    for order in sorted(orders):
        lines.append(f'  - ["{order}", "order total", "{fmt(orders[order])}"]')
        fams = sorted((f for (o, f) in counts if o == order and f != NO_FAMILY))
        for fam in fams:
            lines.append(f'  - ["", "{fam}", "{fmt(counts[(order, fam)])}"]')
        if (order, NO_FAMILY) in counts:
            lines.append(f'  - ["", "{NO_FAMILY}", "{fmt(counts[(order, NO_FAMILY)])}"]')
    lines.append(f'  - ["Total", "", "{fmt(total)}"]')
    return "\n".join(lines) + "\n"


def main() -> int:
    utf8_stdout()
    counts, total, n_samples = count()
    text = build(counts, total)
    n_fam = sum(1 for (_, f) in counts if f != NO_FAMILY)
    n_ord = len({o for (o, _) in counts})
    print(f"samples {n_samples}, specimens {total}, orders {n_ord}, families {n_fam}, "
          f"without family {sum(n for (_, f), n in counts.items() if f == NO_FAMILY)}")
    ok = total == EXPECTED_TOTAL and n_samples == EXPECTED_SAMPLES
    print(f"check against the main analysis ({EXPECTED_SAMPLES} samples, {EXPECTED_TOTAL}): "
          f"{'ok' if ok else 'MISMATCH'}")
    if "--check" in sys.argv[1:]:
        same = OUT.exists() and OUT.read_bytes().replace(b"\r\n", b"\n") == text.encode("utf-8")
        print(f"{OUT.name}: {'unchanged' if same else 'DRIFT'}")
        return 0 if ok and same else 1
    if not ok:
        return 1
    OUT.write_bytes(text.encode("utf-8"))
    print(f"written {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
