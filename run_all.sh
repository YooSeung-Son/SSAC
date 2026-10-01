#!/usr/bin/env bash
# Full pipeline: raw data -> results (out/*.json) -> Figure 1 -> abstract PDF.  About 5 minutes on one machine.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
mkdir -p out
$PY src/prep.py            # Elo, rolling serve/return strength, long player-match table (Sackmann)
$PY src/mcp_prep.py        # charted points: Markov leverage per point; matches: Markov expectation; player ids
$PY src/mcp_clutch.py      # R1 big points vs the rest; R5 raw clutch with permutation null
$PY src/mcp_clutch2.py     # R5 after removing tour-wide score effects; simulated null; UPR+ player-match table
$PY src/sack_markov.py     # points-based Markov win expectation for every Sackmann match with stats
$PY src/panel.py           # R2, R3, R6, R7, R8, R9
$PY src/r4_points.py       # R4 break-point / tiebreak-point prediction
$PY src/table_fig_v3.py    # Table 1 numbers; adjusted quintiles for Figure 1
$PY src/r7_robustness.py   # robustness numbers quoted in the abstract
$PY src/fig_quint.py 173.81 134.76   # Figure 1 at the size LaTeX reports (FIGBOX line in latex/main.log)
if command -v tectonic >/dev/null; then (cd latex && tectonic main.tex); elif command -v pdflatex >/dev/null; then (cd latex && pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null); fi
echo done
