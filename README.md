# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/kobe258174/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/kobe258174/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts, hour by hour, how many people watch Grand Theft Auto VI live on Twitch over the 24 UTC hours of its release day (2026-11-19), from everything recorded up to the end of its first hour, as five quantiles an hour (0.05, 0.25, 0.5, 0.75, 0.95). One Python file, standard library only, no network, no model calls: it runs offline in the grader's sandbox.

## Method

k-nearest-launch ratio quantiles. A launch day's audience curve is mostly a matter of when the game unlocked and how big it already was at midnight UTC: across the 38 past launches in the case file, log(hour-1 viewers) and the day's peak ratio correlate at -0.94. Small hour-1 prints mean the unlock came later in the day and the curve explodes (up to 2000x); big hour-1 prints mean the launch was already live at midnight and the curve stays near flat (1-3x). A single global ratio curve fits neither regime, so for each hour the forecast weights every comparable by how close its log hour-1 viewers are to the target's (Gaussian kernel, bandwidth 1.0 in log units) and reports the weighted quantiles of their viewers(h)/viewers(1) ratios, scaled by the target's hour-1 print. The spread is the real cross-launch spread, not an assumed one. Fallback: fewer than 5 donors for an hour uses the house persistence baseline.

Leave-one-out validation on the 38 comparables (pinball loss on the log scale, the contest's own score): persistence baseline 0.93, global ratio quantiles 0.71, this model 0.41 overall and 0.26 on launches with hour-1 >= 20,000 viewers (the regime GTA VI, a console rolling-midnight launch, falls in). The house baseline scores 0.387 on the four practice cases.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` in the input directory and writes `<id>.out` (23 rows, hours 2..24, five quantiles each) to the output directory. Requires Python 3.13, standard library only.

---

<sub>Entered on [make.fun](https://make.fun/?ref=x9f1j2w7&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
