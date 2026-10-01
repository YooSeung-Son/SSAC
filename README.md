# Beyond Under Pressure

Code and manuscript for our MIT Sloan Sports Analytics Conference 2027 abstract, *Beyond Under Pressure: Measuring
Context-Adjusted Clutch Skill in Men's Professional Tennis*. The abstract itself is in `latex/main.pdf`.

## What we did

The ATP's Under Pressure Rating adds up four percentages: break points saved, break points converted, tiebreaks won
and deciding sets won. We wanted to know whether that number actually captures clutch skill, and whether a measure
built from point-level data could do better.

Using the Match Charting Project's point-by-point records, we gave every point a leverage value from a Markov model
of the score, i.e. how much winning that point changes the chance of winning the match. Our measure, UPR+, compares a
player's results on high-leverage points with his own serve and return level in the same match. Score effects that
apply to everyone on tour (servers doing a bit worse on break points, for example) are removed first, and estimates
from small samples are shrunk toward zero.

We then checked whether UPR+ is a stable property of a player, by splitting each player's matches into odd and even
halves and comparing against outcomes simulated with no clutch skill at all. We ran the same reliability checks on the
four official components. Finally, we measured UPR+ over three seasons and asked whether it predicts the next one: wins
beyond what a player's points won would predict, using every tour, Challenger and qualifying match in Jeff Sackmann's
data, and the change in ATP ranking. These models control for share of points won, ranking, age and the number of
charted matches, and we fit them with the official rating as well for comparison.

## Layout

```
src/                 analysis scripts (run in the order listed in run_all.sh)
scripts/get_data.sh  downloads the raw data
run_all.sh           rebuilds every number, the figure and the PDF
latex/               abstract, table, figure, and the side-by-side figure/table block (figtab.tex)
out/                 JSON summaries written by the scripts
```

## Reproducing

```bash
pip install -r requirements.txt
scripts/get_data.sh     # about 450 MB
./run_all.sh            # a few minutes
```

`run_all.sh` builds the PDF with `tectonic` or `pdflatex` if either is installed. Figure 1 is drawn at the exact size
of its slot in the layout. If the LaTeX template changes, look for the `FIGBOX:` line in `latex/main.log` and redraw
with `python src/fig_quint.py <width_pt> <height_pt>`; the figure and table stay the same height either way.

## Data

- Match Charting Project, men's matches and points, from the public GitHub release (7,518 matches when we pulled it;
  the project website lists more).
- Jeff Sackmann's ATP match, ranking and player files. The original `tennis_atp` repository is offline, so we use the
  mirror at `Aneeshers/tennis-sackmann-archive`.

We did not restrict the sample to players who are still active, since dropping players who declined and retired would
make the forward-looking results look better than they are.

Both datasets are licensed CC BY-NC-SA 4.0, so the raw files are not included here and anything derived from them
falls under the same terms.

- *Crowdsourced shot-by-shot professional tennis data* by The Tennis Abstract Match Charting Project,
  http://www.tennisabstract.com/charting/meta.html
- ATP match and ranking data by Jeff Sackmann / Tennis Abstract
