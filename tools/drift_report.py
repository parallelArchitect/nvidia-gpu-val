#!/usr/bin/env python3
"""
Author: Joe McLaren
Project: pascal-gpu-val
File: tools/drift_report.py

Description:
  Generate a forensic "drift report" from PascalVal run history CSV (e.g. train_300.csv).
  This tool is meant for internal analysis/training (not necessarily shipped).

  Core principles:
    - No single-run panic flags
    - Robust statistics (median + MAD)
    - Drift is judged over a WINDOW vs a BASELINE
    - Context correlation is reported when columns exist (pcie, memory, verdicts)

Usage:
  python3 tools/drift_report.py \
    --csv sandbox_new/data/train_300.csv \
    --out sandbox_new/data/drift_report.json \
    --current-window 60 \
    --min-persist 10

  # If you want a separate baseline segment (recommended):
  python3 tools/drift_report.py \
    --csv sandbox_new/data/train_300.csv \
    --out sandbox_new/data/drift_report.json \
    --baseline-rows 220 \
    --current-window 60

Timestamp:
  Generated at runtime (local time zone) inside the output JSON.

Notes:
  - Requires only Python stdlib.
  - Designed to be "code-zoo safe": one file, deterministic outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Robust stats utilities
# -----------------------------

def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def median(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    if not xs:
        return None
    return statistics.median(xs)


def mad(xs: List[float], med: Optional[float] = None) -> Optional[float]:
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    if not xs:
        return None
    if med is None:
        med = statistics.median(xs)
    dev = [abs(x - med) for x in xs]
    return statistics.median(dev) if dev else None


def robust_z(x: float, med: float, mad_val: float, eps: float = 1e-12) -> float:
    # Scale factor 1.4826 makes MAD comparable to std dev for normal distributions
    denom = 1.4826 * max(mad_val, eps)
    return (x - med) / denom


def pct_change(current: float, baseline: float, eps: float = 1e-12) -> float:
    return 100.0 * (current - baseline) / max(abs(baseline), eps)


def safe_mode(values: List[str]) -> Optional[str]:
    values = [v for v in values if v is not None and str(v).strip() != ""]
    if not values:
        return None
    # deterministic: sort by (-count, value)
    counts: Dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return items[0][0]


# -----------------------------
# Data model
# -----------------------------

@dataclass
class SegmentStats:
    n: int
    gflops_med: Optional[float]
    gflops_mad: Optional[float]
    ratio_med: Optional[float]
    ratio_mad: Optional[float]
    time_ms_med: Optional[float]
    time_ms_mad: Optional[float]


@dataclass
class DriftDecision:
    state: str          # STABLE / DRIFTING / UNSTABLE / FAULTED / INSUFFICIENT_DATA
    confidence: float   # 0..1
    rationale: List[str]


# -----------------------------
# Core analysis
# -----------------------------

def compute_segment_stats(rows: List[Dict[str, Any]]) -> SegmentStats:
    g = [_to_float(r.get("gflops")) for r in rows]
    rto = [_to_float(r.get("theoretical_ratio")) for r in rows]
    tms = [_to_float(r.get("time_ms")) for r in rows]

    g_med = median([x for x in g if x is not None])
    g_mad = mad([x for x in g if x is not None], g_med) if g_med is not None else None

    r_med = median([x for x in rto if x is not None])
    r_mad = mad([x for x in rto if x is not None], r_med) if r_med is not None else None

    t_med = median([x for x in tms if x is not None])
    t_mad = mad([x for x in tms if x is not None], t_med) if t_med is not None else None

    return SegmentStats(
        n=len(rows),
        gflops_med=g_med,
        gflops_mad=g_mad,
        ratio_med=r_med,
        ratio_mad=r_mad,
        time_ms_med=t_med,
        time_ms_mad=t_mad,
    )


def assess_persistence(
    rows: List[Dict[str, Any]],
    baseline: SegmentStats,
    z_thresh: float,
    min_persist: int,
) -> Tuple[int, int, List[Tuple[str, float, float]]]:
    """
    Count how many runs in CURRENT window are "low performance" vs baseline
    using robust z on gflops and ratio.

    Returns:
      low_count, considered_count, samples(list of (rid, z_gflops, z_ratio))
    """
    samples: List[Tuple[str, float, float]] = []

    if baseline.gflops_med is None or baseline.gflops_mad is None:
        return (0, 0, samples)
    if baseline.ratio_med is None or baseline.ratio_mad is None:
        # ratio may be missing; that's OK, we can still use gflops
        pass

    considered = 0
    low = 0

    for r in rows:
        rid = str(r.get("rid", "")).strip()
        g = _to_float(r.get("gflops"))
        ratio = _to_float(r.get("theoretical_ratio"))

        if g is None or not math.isfinite(g):
            continue

        zg = robust_z(g, baseline.gflops_med, baseline.gflops_mad)
        zr = 0.0
        zr_valid = False

        if baseline.ratio_med is not None and baseline.ratio_mad is not None and ratio is not None and math.isfinite(ratio):
            zr = robust_z(ratio, baseline.ratio_med, baseline.ratio_mad)
            zr_valid = True

        considered += 1
        # "low" means significantly below baseline
        # If ratio exists, require both (more conservative). If ratio absent, use gflops alone.
        is_low = (zg <= -abs(z_thresh)) and ((not zr_valid) or (zr <= -abs(z_thresh)))
        if is_low:
            low += 1

        if rid:
            samples.append((rid, zg, zr if zr_valid else float("nan")))

    # Keep only the worst few for report
    samples.sort(key=lambda x: x[1])  # most negative zg first
    samples = samples[:max(min_persist, 10)]
    return (low, considered, samples)


def correlate_context(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Lightweight correlation summary:
      - PCIe stability: gen/width mode + unique count
      - Memory (UM) ratio median + spread if present
      - Verdict tallies
    """
    pcie_gen = [str(r.get("pcie_gen", "")).strip() for r in rows]
    pcie_width = [str(r.get("pcie_width", "")).strip() for r in rows]
    mem_ratio = [_to_float(r.get("um_ratio")) for r in rows]
    um_gib = [_to_float(r.get("um_gib")) for r in rows]

    finals = [str(r.get("final", "")).strip() for r in rows]
    compute_verdicts = [str(r.get("compute_verdict", "")).strip() for r in rows]
    mem_verdicts = [str(r.get("mem_verdict", "")).strip() for r in rows]
    pcie_verdicts = [str(r.get("pcie_verdict", "")).strip() for r in rows]

    def tally(vals: List[str]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for v in vals:
            if not v:
                continue
            out[v] = out.get(v, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))

    gen_mode = safe_mode(pcie_gen)
    width_mode = safe_mode(pcie_width)

    gen_unique = sorted({v for v in pcie_gen if v})
    width_unique = sorted({v for v in pcie_width if v})

    mem_ratio_vals = [x for x in mem_ratio if x is not None and math.isfinite(x)]
    um_gib_vals = [x for x in um_gib if x is not None and math.isfinite(x)]

    ctx: Dict[str, Any] = {
        "pcie": {
            "gen_mode": gen_mode,
            "width_mode": width_mode,
            "gen_unique": gen_unique,
            "width_unique": width_unique,
            "stable": (len(gen_unique) <= 1 and len(width_unique) <= 1),
        },
        "memory": {
            "um_ratio_median": median(mem_ratio_vals) if mem_ratio_vals else None,
            "um_ratio_mad": mad(mem_ratio_vals, median(mem_ratio_vals)) if mem_ratio_vals else None,
            "um_gib_median": median(um_gib_vals) if um_gib_vals else None,
        },
        "verdicts": {
            "final": tally(finals),
            "compute": tally(compute_verdicts),
            "memory": tally(mem_verdicts),
            "pcie": tally(pcie_verdicts),
        },
    }
    return ctx


def decide_state(
    baseline: SegmentStats,
    current: SegmentStats,
    low_count: int,
    considered: int,
    min_persist: int,
    z_thresh: float,
) -> DriftDecision:
    rationale: List[str] = []

    # Hard insufficient data
    if (
        baseline.gflops_med is None
        or baseline.gflops_mad is None
        or current.gflops_med is None
        or current.gflops_mad is None
        or considered == 0
    ):
        return DriftDecision("INSUFFICIENT_DATA", 0.35, ["Not enough numeric compute samples to assess drift."])

    # Drift magnitude (median shift)
    dg = pct_change(current.gflops_med, baseline.gflops_med)
    rationale.append(f"GFLOPS median shift: {dg:+.2f}% (baseline={baseline.gflops_med:.2f}, current={current.gflops_med:.2f})")

    # Variance change (MAD)
    # Use ratio of MADs; if baseline MAD tiny, protect with epsilon.
    eps = 1e-9
    var_ratio = (current.gflops_mad + eps) / (baseline.gflops_mad + eps)
    rationale.append(f"GFLOPS MAD ratio: {var_ratio:.2f}x (baseline_mad={baseline.gflops_mad:.3f}, current_mad={current.gflops_mad:.3f})")

    # Persistence criterion
    rationale.append(f"Low-perf persistence: {low_count}/{considered} runs below -{z_thresh:.1f} robust-z; min_persist={min_persist}")

    # State machine (conservative, time-aware)
    # - STABLE: low_count < min_persist AND |dg| small AND variance not exploding
    # - DRIFTING: persistence met OR median shift notable
    # - UNSTABLE: persistence met AND variance ratio high
    # - FAULTED: reserved for explicit error signals (not implemented here; handled by caller if desired)
    #
    # NOTE: thresholds are intentionally mild; you tune them with experience.
    dg_abs = abs(dg)
    variance_exploding = var_ratio >= 2.0
    persistence = low_count >= min_persist

    # Confidence is evidence-weighted: more considered samples and stronger effects => higher confidence.
    # Keep bounded and honest.
    effect = min(1.0, (dg_abs / 5.0) + (math.log(var_ratio + 1e-9) / 2.0) + (low_count / max(considered, 1)))
    sample_factor = min(1.0, considered / 50.0)
    conf = max(0.50, min(0.97, 0.45 + 0.35 * sample_factor + 0.25 * effect))

    if persistence and variance_exploding:
        return DriftDecision("UNSTABLE", conf, rationale + ["Persistent low performance with increased variance (instability signature)."])
    if persistence or dg_abs >= 4.0:
        # 4% median shift is meaningful in your tight-band dataset
        return DriftDecision("DRIFTING", conf, rationale + ["Statistically meaningful deviation vs baseline (drift signature)."])
    if variance_exploding:
        return DriftDecision("DRIFTING", max(0.65, conf - 0.05), rationale + ["Variance increased vs baseline (possible drift)."])
    return DriftDecision("STABLE", max(0.70, conf - 0.10), rationale + ["Within learned envelope; no persistent low-performance cluster detected."])


# -----------------------------
# IO + CLI
# -----------------------------

def load_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        rows = [dict(row) for row in r]
    # Sort by RID (lexicographic time order) if present
    rows.sort(key=lambda x: str(x.get("rid", "")).strip())
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a robust drift report from PascalVal run history CSV.")
    ap.add_argument("--csv", required=True, help="Input CSV (e.g., sandbox_new/data/train_300.csv)")
    ap.add_argument("--out", required=True, help="Output JSON report path")
    ap.add_argument("--current-window", type=int, default=60, help="Number of most recent runs to assess as 'current'")
    ap.add_argument("--baseline-rows", type=int, default=0,
                    help="If >0, use first N rows as baseline. Otherwise baseline = all rows excluding current-window.")
    ap.add_argument("--z-thresh", type=float, default=3.0, help="Robust-z threshold for 'low performance' runs")
    ap.add_argument("--min-persist", type=int, default=10, help="Minimum low runs in current-window before calling drift")
    ap.add_argument("--emit-worst", type=int, default=10, help="How many worst samples to include (by gflops z)")
    args = ap.parse_args()

    rows = load_csv(args.csv)
    n = len(rows)
    if n < max(args.current_window + 5, 20):
        report = {
            "schema": "pascalval.drift_report.v1",
            "generated_at": datetime.now().astimezone().isoformat(),
            "input": {"csv": args.csv, "rows": n},
            "decision": {"state": "INSUFFICIENT_DATA", "confidence": 0.35},
            "note": "Not enough rows to compute baseline vs current with robustness.",
        }
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, sort_keys=False)
        print(args.out)
        return 0

    current_window = max(5, min(args.current_window, n))
    current_rows = rows[-current_window:]

    if args.baseline_rows and args.baseline_rows > 0:
        bN = min(args.baseline_rows, n - current_window)
        baseline_rows = rows[:bN]
    else:
        baseline_rows = rows[: max(1, n - current_window)]

    baseline = compute_segment_stats(baseline_rows)
    current = compute_segment_stats(current_rows)

    low_count, considered, worst_samples = assess_persistence(
        current_rows, baseline, z_thresh=args.z_thresh, min_persist=args.min_persist
    )

    decision = decide_state(
        baseline=baseline,
        current=current,
        low_count=low_count,
        considered=considered,
        min_persist=args.min_persist,
        z_thresh=args.z_thresh,
    )

    # Context correlation summary on current window
    ctx = correlate_context(current_rows)

    # Trim worst samples for report readability
    worst_trim = worst_samples[: max(1, args.emit_worst)]
    worst_out = []
    for rid, zg, zr in worst_trim:
        worst_out.append({
            "rid": rid,
            "z_gflops": zg,
            "z_ratio": None if (zr != zr) else zr,  # NaN -> None
        })

    # Optional: if verdicts indicate explicit failures, elevate to FAULTED
    verdicts = ctx.get("verdicts", {})
    final_tally = verdicts.get("final", {}) if isinstance(verdicts, dict) else {}
    has_fail = any(k.upper() in ("FAIL", "FAILED") for k in final_tally.keys())
    if has_fail and decision.state in ("STABLE", "DRIFTING", "UNSTABLE"):
        decision = DriftDecision(
            state="FAULTED",
            confidence=min(0.99, max(0.85, decision.confidence + 0.10)),
            rationale=decision.rationale + ["Final verdict includes FAIL; elevating state to FAULTED."],
        )

    newest = str(rows[-1].get("rid", "")).strip()
    oldest = str(rows[0].get("rid", "")).strip()
    current_newest = str(current_rows[-1].get("rid", "")).strip()
    current_oldest = str(current_rows[0].get("rid", "")).strip()

    report: Dict[str, Any] = {
        "schema": "pascalval.drift_report.v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "input": {
            "csv": args.csv,
            "rows_total": n,
            "rid_range_total": {"oldest": oldest, "newest": newest},
            "baseline_rows": len(baseline_rows),
            "current_window": current_window,
            "current_rid_range": {"oldest": current_oldest, "newest": current_newest},
            "params": {
                "z_thresh": args.z_thresh,
                "min_persist": args.min_persist,
            },
        },
        "baseline": {
            "n": baseline.n,
            "gflops_median": baseline.gflops_med,
            "gflops_mad": baseline.gflops_mad,
            "ratio_median": baseline.ratio_med,
            "ratio_mad": baseline.ratio_mad,
            "time_ms_median": baseline.time_ms_med,
            "time_ms_mad": baseline.time_ms_mad,
        },
        "current": {
            "n": current.n,
            "gflops_median": current.gflops_med,
            "gflops_mad": current.gflops_mad,
            "ratio_median": current.ratio_med,
            "ratio_mad": current.ratio_mad,
            "time_ms_median": current.time_ms_med,
            "time_ms_mad": current.time_ms_mad,
        },
        "delta": {
            "gflops_median_pct": None if (baseline.gflops_med is None or current.gflops_med is None)
                                else pct_change(current.gflops_med, baseline.gflops_med),
            "ratio_median_pct": None if (baseline.ratio_med is None or current.ratio_med is None)
                               else pct_change(current.ratio_med, baseline.ratio_med),
            "gflops_mad_ratio": None if (baseline.gflops_mad is None or current.gflops_mad is None)
                               else (current.gflops_mad + 1e-9) / (baseline.gflops_mad + 1e-9),
        },
        "persistence": {
            "low_count": low_count,
            "considered": considered,
            "worst_samples": worst_out,
        },
        "context": ctx,
        "decision": {
            "state": decision.state,
            "confidence": decision.confidence,
            "rationale": decision.rationale,
        },
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, sort_keys=False)

    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
