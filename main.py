"""GTA VI launch-day Twitch CCU forecaster (gta6-twitch-ccu-forecast-01).

Predicts, for each hour 2..24 of GTA VI's release day (UTC, 2026-11-19), the
concurrent Twitch viewers in the "Grand Theft Auto VI" category, as five
quantiles (0.05, 0.25, 0.5, 0.75, 0.95), from the state recorded up to the
end of hour 1 (that hour's print included).

Method: kernel-weighted ratio quantiles.
  A launch day's audience curve is mostly a matter of when the game unlocked
  and how big it already was at midnight UTC. Across the 38 past launches in
  the case file, log(hour-1 viewers) and the day's peak ratio correlate at
  -0.94: small hour-1 prints mean the unlock came later in the day and the
  curve explodes (up to 2000x); big hour-1 prints mean the launch was already
  live at midnight and the curve stays near flat (1-3x). A single global
  ratio curve fits neither regime, so for each hour the forecast weights
  every comparable by how close its log hour-1 viewers are to the target's
  (Gaussian kernel, bandwidth 1.0 in log units) and reports the weighted
  quantiles of their viewers(h)/viewers(1) ratios, scaled by the target's
  hour-1 print. The spread is the real cross-launch spread, not an assumed
  one.

  Leave-one-out validation on the 38 comparables in the kit (pinball loss on
  the log scale, the contest's own score): persistence baseline 0.93, global
  ratio quantiles 0.71, this model 0.41 overall and 0.26 on launches with
  hour-1 >= 20,000 viewers (the regime GTA VI, a console rolling-midnight
  launch, falls in). The house baseline scores 0.387 on the four practice
  cases; see the kit's grade-local run for this entry's practice score.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only: no network, no pip packages, no
model calls. -P keeps this file's directory OFF sys.path, so a module shipped
beside main.py imports only after the sys.path line below.

What an entry must do:
  - read every <id>.in in the input directory (the grader runs one case at a
    time, with only that case's file there): JSON
    {"schema": "makefun/twitch-ccu-case/v1", "id", "game", "day_start",
    "checkpoint", "levels", "known", "forecast", "history",
    "history_hourly", "streams", "categories_hourly", "comparables",
    "notes"}, where `known` holds hour 1's print {hour: 1, from, until,
    viewers, channels, reads} and `forecast` lists the hours to forecast,
    2..24;
  - write <id>.out to the output directory: {"hours": [{"hour": 2, "q":
    [q05, q25, q50, q75, q95]}, ...]}, exactly 23 rows, hours 2..24 each
    once, no other key, each q five finite numbers above 0 and at most 1e9
    (viewers; fractions allowed), non-decreasing; at most 65,536 bytes.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level (persistence fallback only).
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# The spread of an hour's move, in natural-log units per square-root hour
# (persistence fallback only, same as the house baseline).
SIGMA = 0.15
# Donors taken per hour: every comparable is weighted by closeness in log
# hour-1 viewers (Gaussian kernel below); fewer than this many donors with
# both hour 1 and hour h falls back to the persistence baseline.
MIN_DONORS = 5
# Bandwidth, in natural-log units, of the Gaussian kernel over log hour-1
# viewers: a donor 1.0 log units away carries weight exp(-0.5) ~ 0.61.
KERNEL_BW = 1.0


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100 rounds it.

    A whole number comes back as an int, so json.dumps writes 1240 where JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def _parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def day_hours(comp: dict) -> dict[int, float]:
    """A comparable's release-day hourly viewers, keyed by hour 1..24.

    Only positive viewer counts count as observations: a zero is a sampling
    gap in the source data, not a real zero for a launch of this size.
    """
    day_start = _parse_ts(comp["day_start"])
    out: dict[int, float] = {}
    for point in comp.get("hourly") or []:
        viewers = point.get("viewers")
        if viewers is None or viewers <= 0:
            continue
        delta_h = (_parse_ts(point["from"]) - day_start).total_seconds() / 3600
        if 0 <= delta_h < 24 and abs(delta_h - round(delta_h)) < 1e-9:
            out[int(round(delta_h)) + 1] = float(viewers)
    return out


def persistence(p1: float, h: int) -> list[float | int]:
    """The house baseline: every hour stays at hour 1's print, spread widening like a random walk's in log space."""
    return [round2(max(1.0, p1 * math.exp(z * SIGMA * math.sqrt(h - 1)))) for z in Z]


def weighted_quantile(pairs: list[tuple[float, float]], tau: float) -> float:
    """Quantile of (value, weight) pairs, sorted by value, interpolating in cumulative weight.

    The running weight reaches tau of the total weight between two values;
    the result is the linear interpolation between them at that point.
    """
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    target = tau * total
    acc = 0.0
    for i, (value, weight) in enumerate(pairs):
        acc += weight
        if acc >= target:
            if i == 0:
                return value
            prev_value, _ = pairs[i - 1]
            frac = 1 - (acc - target) / weight if weight > 0 else 0.0
            return prev_value * (1 - frac) + value * frac
    return pairs[-1][0]


def hour_quantiles(p1: float, donors: list[dict[int, float]], h: int) -> list[float | int]:
    """The forecast row for hour h: kernel-weighted ratio quantiles, with a persistence fallback.

    Every comparable with both hour 1 and hour h contributes its ratio
    viewers(h)/viewers(1), weighted by a Gaussian kernel over the distance
    between its log hour-1 viewers and the target's. Fewer than MIN_DONORS
    donors for the hour falls back to the persistence baseline. The pairs
    are sorted by ratio before the weighted quantile, and the donors arrive
    in the case file's order, so two runs write the same bytes.
    """
    pairs: list[tuple[float, float]] = []
    for dh in donors:
        if 1 in dh and h in dh and dh[1] > 0:
            distance = abs(math.log(dh[1]) - math.log(p1))
            weight = math.exp(-0.5 * (distance / KERNEL_BW) ** 2)
            pairs.append((dh[h] / dh[1], weight))
    if len(pairs) < MIN_DONORS:
        return persistence(p1, h)
    return [round2(min(1e9, max(1.0, p1 * weighted_quantile(pairs, tau)))) for tau in LEVELS]


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    donors = [day_hours(c) for c in case.get("comparables") or []]
    return {"hours": [{"hour": row["hour"], "q": hour_quantiles(p1, donors, row["hour"])} for row in case["forecast"]]}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
