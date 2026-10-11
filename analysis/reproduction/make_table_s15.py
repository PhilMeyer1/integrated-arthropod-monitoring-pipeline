"""Table S15: key results of the main analysis (53 samples, sample 180 excluded) next to the
sensitivity analysis (54 samples, sample 180 included).

Reads key_numbers.csv of both runs written by run_all.sh (key_numbers.R); every cell is
computed from the unrounded run values, no number is typed in. Percentages are
100 x (exp(b) - 1) of the model coefficients ("per mm decrease": 100 x (exp(-b) - 1)).
Standard library only.

Usage (from the repository root, after bash analysis/reproduction/run_all.sh <OUT>):
    python analysis/reproduction/make_table_s15.py <OUT>/main_without_180 <OUT>/sensitivity_with_180 [--check]
Writes analysis/reproduction/tables/table_s15_sensitivity_sample180.yaml; --check compares the
generated table with that file instead of writing it (exit 1 on any difference).
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "tables" / "table_s15_sensitivity_sample180.yaml"
# key_numbers.csv of earlier runs used German key names (textzahlen.csv)
ALIASES = {"proben_n": "samples_n", "ind_mittlere_groesse_mm": "ind_mean_size_mm"}

HEAD = '''title: "Sensitivity analysis: main results without and with sample 180."
columns:
  - name: "Quantity"
    width: 5.5
    align: left
  - name: "Table"
    width: 1.2
    align: center
  - name: "Main analysis (53 samples, without sample 180)"
    width: 3.0
    align: center
  - name: "Sensitivity analysis (54 samples, with sample 180)"
    width: 3.0
    align: center
data:
'''
TAIL = ('note: "Sample 180 (site LA2, 2024) was collected over two days instead of 24 hours; the '
        'main analysis (Tables S4-S12, main text) excludes it, and this table repeats the key '
        'results with it. Percentages are 100 x (exp(b) - 1) '
        'of the model coefficients. P-values of permutation tests (S8, S10, S12) vary slightly '
        'between runs because of random permutations."\n'
        'keep_together: true\n')


def read(run: Path) -> dict[str, float]:
    path = run / "key_numbers.csv"
    if not path.exists():
        path = run / "textzahlen.csv"
    with path.open(encoding="utf-8") as f:
        return {ALIASES.get(r["key"], r["key"]): float(r["value"]) if r["value"] not in ("", "NA") else math.nan
                for r in csv.DictReader(f)}


def pct(b: float) -> float:
    return 100 * (math.exp(b) - 1)


def rows(k: dict[str, float]) -> list[tuple[str, str, str]]:
    p3 = lambda key: f"{k[key]:.3f}"  # noqa: E731
    eff = lambda m: f"{pct(k[m + '_treatmenteco_b']):+.1f} ({k[m + '_treatmenteco_p']:.3f})"  # noqa: E731
    return [
        ("Samples", "S3", f"{k['samples_n']:.0f}"),
        ("Sampling units per model (diversity and abundance / biomass)", "S4, S6, S7",
         f"{k['div_N']:.0f} / {k['bio_N']:.0f}"),
        ("Taxa (family or order x size fraction)", "S12", f"{k['taxa_n']:.0f}"),
        ("Abundance, hedgerow effect (%, P)", "S6", eff("abu")),
        ("Abundance, post-hoc contrast control - eco (P)", "S5", p3("abu_posthoc_p")),
        ("Abundance, size x hedgerow interaction (P)", "S5, S6", p3("abu_size_treatmenteco_p")),
        ("Abundance, interaction (% per mm decrease in body size)", "S6",
         f"{pct(-k['abu_size_treatmenteco_b']):.1f}"),
        ("Diversity, hedgerow effect (%, P)", "S4", eff("div")),
        ("Biomass, hedgerow effect (%, P)", "S7", eff("bio")),
        ("Temperature effect per °C (diversity / abundance / biomass, %)", "S4, S6, S7",
         " / ".join(f"{pct(k[m + '_temp_mean_b']):.2f}" for m in ("div", "abu", "bio"))),
        ("Diversity, day of year (P)", "S4", p3("div_yday_p")),
        ("PERMANOVA, hedgerow R² (%, P)", "S8",
         f"{100 * k['permanova_treatment_R2']:.2f} ({k['permanova_treatment_p']:.3f})"),
        ("PERMDISP, hedgerow (P)", "S9", p3("permdisp_p")),
        ("Beta diversity, total dissimilarity (P)", "S10", p3("beta_Total_p")),
        ("Beta diversity, nestedness (P)", "S10", p3("beta_Nestedness_p")),
        ("Beta diversity, turnover (P)", "S10", p3("beta_Turnover_p")),
        ("Beta dispersion, total / turnover / nestedness (P)", "S11",
         " / ".join(p3(f"betadisp_{m}_p") for m in ("Total", "Turnover", "Nestedness"))),
        ("Indicator taxa with P < 0.1 (of which control-associated)", "S12",
         f"{k['ind_n']:.0f} ({k['ind_control_n']:.0f})"),
        ("Mean body size of the indicator taxa (mm)", "S12", f"{k['ind_mean_size_mm']:.2f}"),
    ]


def build(main: dict[str, float], sens: dict[str, float]) -> str:
    lines = [f'  - ["{a}", "{t}", "{m}", "{s}"]'
             for (a, t, m), (_, _, s) in zip(rows(main), rows(sens))]
    return HEAD + "\n".join(lines) + "\n" + TAIL


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    if len(args) != 2:
        print(__doc__)
        return 2
    main_run, sens_run = read(Path(args[0])), read(Path(args[1]))
    ok = main_run["samples_n"] == 53 and sens_run["samples_n"] == 54
    print(f"samples: main {main_run['samples_n']:.0f}, sensitivity {sens_run['samples_n']:.0f} "
          f"({'ok' if ok else 'MISMATCH, expected 53 and 54'})")
    text = build(main_run, sens_run)
    if "--check" in argv:
        same = OUT.exists() and OUT.read_bytes().replace(b"\r\n", b"\n") == text.encode("utf-8")
        print(f"{OUT.name}: {'unchanged' if same else 'DRIFT'}")
        if not same:
            import difflib
            old = OUT.read_bytes().decode("utf-8").replace("\r\n", "\n").splitlines()
            sys.stdout.writelines(l + "\n" for l in difflib.unified_diff(old, text.splitlines(), lineterm=""))
        return 0 if ok and same else 1
    if not ok:
        return 1
    OUT.write_bytes(text.encode("utf-8"))
    print(f"written {OUT}")
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    sys.exit(main(sys.argv[1:]))
