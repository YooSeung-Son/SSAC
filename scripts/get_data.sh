#!/usr/bin/env bash
# Download the raw data (not redistributed in this repo; both sources are CC BY-NC-SA 4.0).
#   data/tennis_atp/  Jeff Sackmann ATP matches, rankings and players, via the archive mirror
#                     github.com/Aneeshers/tennis-sackmann-archive (the original tennis_atp repo is offline)
#   data/mcp/         Match Charting Project, men's matches and points (github.com/JeffSackmann/tennis_MatchChartingProject)
set -euo pipefail
cd "$(dirname "$0")/.."
ATP=https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main/atp
MCP=https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master
mkdir -p data/tennis_atp data/mcp

get() { [ -s "$2" ] || curl -fsSL --retry 3 -o "$2" "$1"; }

for y in $(seq 2000 2026); do
  get "$ATP/atp_matches_${y}.csv" "data/tennis_atp/atp_matches_${y}.csv"
  get "$ATP/atp_matches_qual_chall_${y}.csv" "data/tennis_atp/atp_matches_qual_chall_${y}.csv"
done
for f in atp_rankings_10s.csv atp_rankings_20s.csv atp_rankings_current.csv atp_players.csv; do
  get "$ATP/$f" "data/tennis_atp/$f"
done
for f in charting-m-matches.csv charting-m-points-to-2009.csv charting-m-points-2010s.csv charting-m-points-2020s.csv; do
  get "$MCP/$f" "data/mcp/$f"
done
echo "data ready: $(ls data/tennis_atp | wc -l) ATP files, $(ls data/mcp | wc -l) MCP files"
