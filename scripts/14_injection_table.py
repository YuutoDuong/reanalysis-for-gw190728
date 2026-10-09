"""Appendix B: the injection table, generated from the results files.

Takes each injection's parameters from injection_lnb.json and its Bayes
factors as the paper quotes them (injections.background: full-grid
likelihood once script 30 has run, mass-cut offset included), and writes
paper_appB_table.tex, so no number in the table is transcribed by hand.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from scalar_env.injections import background, select_sr_subset

campaign = json.loads(Path("injection_lnb.json").read_text())
lnb = background()
subset = {key for key, _ in select_sr_subset("injection_lnb.json")}


def cell(value):
    return f"${value:+.2f}$" if abs(value) < 100 else f"${value:+.0f}$"


lines = []
for key in sorted(campaign):
    p = campaign[key]["parameters"]
    agnostic, sr = lnb[key]
    sr_cell = cell(sr) if key in subset else "---"
    lines.append(f"  {key.replace('inj', '')} & {p['mass_1']:.2f} & {p['mass_2']:.2f} & "
                 f"{p['a_1']:.2f} & {p['luminosity_distance']:.0f} & "
                 f"{p['theta_jn']:.2f} & {cell(agnostic)} & {sr_cell} \\\\")

out = Path("paper_appB_table.tex")
out.write_text("\n".join(lines) + "\n")
print(f"{out}: {len(lines)} rows")
