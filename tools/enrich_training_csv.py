#!/usr/bin/env python3
"""
Author: Joe McLaren
Project: pascal-gpu-val
File: tools/enrich_training_csv.py

Description:
  Internal/training tool: build an enriched, model-ready CSV from PascalVal run.json history.
  Extracts compute outcomes + correlated telemetry (clock/temp/power) + PCIe + memory signals.

  This avoids “code zoo” by:
    - One deterministic pass over run.json files
    - Safe schema navigation (multiple fallback paths)
    - No external deps (stdlib only)
    - Blanks for missing fields (never crashes on schema drift)

Usage:
  # Canonical live runs:
  python3 tools/enrich_training_csv.py \
    --root results/json \
    --out sandbox_new/data/train_300_enriched.csv

  # If you want to constrain to a RID list (e.g., your last 300):
  python3 tools/enrich_training_csv.py \
    --root results/json \
    --rid-list sandbox_new/data/rids_last_300_forensic.txt \
    --out sandbox_new/data/train_300_enriched.csv

  # If using your staged history tree:
  python3 tools/enrich_training_csv.py \
    --root history_results/results/json \
    --out sandbox_new/data/train_300_enriched.csv

Timestamp:
  Printed at runtime, file is deterministic given inputs.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple


# -----------------------------
# Helpers
# -----------------------------

def get_in(doc: Any, path: List[Any], default: Any = None) -> Any:
    cur = doc
    for k in path:
        if isinstance(k, int):
            if not isinstance(cur, list) or k < 0 or k >= len(cur):
                return default
            cur = cur[k]
        else:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
    return cur


def first_present(doc: Any, paths: List[List[Any]], default: Any = None) -> Any:
    for p in paths:
        v = get_in(doc, p, None)
        if v is not None:
            return v
    return default


def to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        try:
            f = float(x)
            return f if math.isfinite(f) else None
        except Exception:
            return None
    s = str(x).strip()
    if not s:
        return None
    try:
        f = float(s)
        return f if math.isfinite(f) else None
    except ValueError:
        return None


def to_int(x: Any) -> Optional[int]:
    f = to_float(x)
    if f is None:
        return None
    try:
        return int(round(f))
    except Exception:
        return None


def fmt_num(x: Optional[float], nd: int = 6) -> str:
    if x is None:
        return ""
    # keep compact but stable
    return f"{x:.{nd}f}".rstrip("0").rstrip(".")


def pct_drop(current: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if current is None or baseline is None or baseline == 0:
        return None
    return 100.0 * (baseline - current) / abs(baseline)


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x)
    return s


def read_rid_list(path: str) -> List[str]:
    rids: List[str] = []
    with open(path, "r") as f:
        for line in f:
            rid = line.strip()
            if rid:
                rids.append(rid)
    return rids


def iter_runjson_paths(root: str, rid_list: Optional[List[str]] = None) -> Iterable[Tuple[str, str]]:
    """
    Yields (rid, runjson_path).
    root is expected like: results/json or history_results/results/json
    """
    root = root.rstrip("/")

    if rid_list:
        for rid in rid_list:
            p = os.path.join(root, rid, "run.json")
            if os.path.isfile(p):
                yield rid, p
        return

    # Discover all run.json
    for p in glob.glob(os.path.join(root, "*", "run.json")):
        rid = os.path.basename(os.path.dirname(p))
        yield rid, p


# -----------------------------
# Extraction schema
# -----------------------------

def extract_row(rid: str, doc: Dict[str, Any]) -> Dict[str, str]:
    # Compute outcomes
    gflops = to_float(first_present(doc, [
        ["results", "compute", "gflops"],
        ["gflops"],
    ]))
    time_ms = to_float(first_present(doc, [
        ["results", "compute", "time_ms"],
        ["time_ms"],
    ]))
    ratio = to_float(first_present(doc, [
        ["results", "compute", "theoretical_ratio"],
        ["theoretical_ratio"],
    ]))

    # Compute metadata
    engine = safe_str(first_present(doc, [
        ["results", "compute", "engine"],
        ["engine"],
    ], ""))
    mode = safe_str(first_present(doc, [
        ["results", "compute", "mode"],
        ["mode"],
    ], ""))
    n = to_int(first_present(doc, [
        ["results", "compute", "n"],
        ["n"],
    ]))
    reps = to_int(first_present(doc, [
        ["results", "compute", "reps"],
        ["reps"],
    ]))

    final = safe_str(first_present(doc, [["final"], ["results", "final"]], ""))

    # PCIe (names can differ across versions; include your known fields)
    pcie_gen = safe_str(first_present(doc, [
        ["results", "pcie", "gen"],
        ["results", "pcie", "generation"],
        ["pcie", "gen"],
        ["pcie", "generation"],
    ], ""))

    pcie_width = safe_str(first_present(doc, [
        ["results", "pcie", "width"],
        ["pcie", "width"],
    ], ""))

    pcie_eff = to_float(first_present(doc, [
        ["results", "pcie", "efficiency_pct"],
        ["pcie", "efficiency_pct"],
    ]))

    tx_gbs = to_float(first_present(doc, [
        ["results", "pcie", "tx_avg_gbs"],
        ["pcie", "tx_avg_gbs"],
    ]))
    rx_gbs = to_float(first_present(doc, [
        ["results", "pcie", "rx_avg_gbs"],
        ["pcie", "rx_avg_gbs"],
    ]))
    comb_gbs = to_float(first_present(doc, [
        ["results", "pcie", "combined_gbs"],
        ["pcie", "combined_gbs"],
    ]))
    pcie_verdict = safe_str(first_present(doc, [["results", "pcie", "verdict"]], ""))

    # Memory / UM
    prefetch_gbs = to_float(first_present(doc, [
        ["results", "memory", "prefetch_gbs"],
        ["results", "memory", "prefetch_gib_s"],  # in case older naming
        ["results", "memory", "prefetch_gbs_s"],
        ["memory", "prefetch_gbs"],
    ]))
    mem_verdict = safe_str(first_present(doc, [["results", "memory", "verdict"]], ""))
    naive_skip_reason = safe_str(first_present(doc, [["results", "memory", "naive_skip_reason"]], ""))

    # Baselines (caps)
    theo_fp32 = to_float(first_present(doc, [["caps", "performance_baseline", "theoretical_fp32_gflops"]], None))
    realistic_base = to_float(first_present(doc, [["caps", "performance_baseline", "realistic_baseline_gflops"]], None))
    healthy_min = to_float(first_present(doc, [["caps", "performance_baseline", "healthy_range_min_gflops"]], None))
    healthy_max = to_float(first_present(doc, [["caps", "performance_baseline", "healthy_range_max_gflops"]], None))

    # Telemetry snapshots (prefer results.compute.telemetry; fall back to gpu_state)
    # Before
    clk_before = to_float(first_present(doc, [
        ["results", "compute", "telemetry", "before_sgemm", "clock_sm_mhz"],
        ["gpu_state", "before_sgemm", "clock_sm_mhz"],
    ]))
    temp_before = to_float(first_present(doc, [
        ["results", "compute", "telemetry", "before_sgemm", "temp_c"],
        ["gpu_state", "before_sgemm", "temp_c"],
    ]))
    pwr_before = to_float(first_present(doc, [
        ["results", "compute", "telemetry", "before_sgemm", "power_w"],
        ["gpu_state", "before_sgemm", "power_w"],
    ]))

    # During
    clk_during = to_float(first_present(doc, [
        ["results", "compute", "telemetry", "during_test", "clock_sm_mhz"],
        ["gpu_state", "during_sgemm", "clock_sm_mhz"],
    ]))
    temp_during = to_float(first_present(doc, [
        ["results", "compute", "telemetry", "during_test", "temp_c"],
        ["gpu_state", "during_sgemm", "temp_c"],
    ]))
    pwr_during = to_float(first_present(doc, [
        ["results", "compute", "telemetry", "during_test", "power_w"],
        ["gpu_state", "during_sgemm", "power_w"],
    ]))

    # After
    clk_after = to_float(first_present(doc, [
        ["results", "compute", "telemetry", "after", "clock_sm_mhz"],
        ["gpu_state", "after_sgemm", "clock_sm_mhz"],
    ]))
    temp_after = to_float(first_present(doc, [
        ["results", "compute", "telemetry", "after", "temp_c"],
        ["gpu_state", "after_sgemm", "temp_c"],
    ]))
    pwr_after = to_float(first_present(doc, [
        ["results", "compute", "telemetry", "after", "power_w"],
        ["gpu_state", "after_sgemm", "power_w"],
    ]))

    # Derived signals
    clk_drop_pct = pct_drop(clk_during, clk_before)  # positive means drop
    temp_rise_c = (temp_during - temp_before) if (temp_during is not None and temp_before is not None) else None
    pwr_rise_w = (pwr_during - pwr_before) if (pwr_during is not None and pwr_before is not None) else None

    # Device identity (keep it simple)
    gpu_name = safe_str(first_present(doc, [
        ["caps", "name"],
        ["name"],
    ], ""))
    uuid = safe_str(first_present(doc, [
        ["gpu_uuid"],
        ["uuid"],
    ], ""))

    row: Dict[str, str] = {
        "rid": rid,
        "final": final,

        # Compute outcome
        "engine": engine,
        "mode": mode,
        "n": safe_str(n) if n is not None else "",
        "reps": safe_str(reps) if reps is not None else "",
        "gflops": fmt_num(gflops, 4),
        "time_ms": fmt_num(time_ms, 6),
        "theoretical_ratio": fmt_num(ratio, 12),

        # Telemetry
        "clock_before_mhz": fmt_num(clk_before, 2),
        "clock_during_mhz": fmt_num(clk_during, 2),
        "clock_after_mhz": fmt_num(clk_after, 2),
        "clock_drop_pct": fmt_num(clk_drop_pct, 4),

        "temp_before_c": fmt_num(temp_before, 2),
        "temp_during_c": fmt_num(temp_during, 2),
        "temp_after_c": fmt_num(temp_after, 2),
        "temp_rise_c": fmt_num(temp_rise_c, 4),

        "power_before_w": fmt_num(pwr_before, 3),
        "power_during_w": fmt_num(pwr_during, 3),
        "power_after_w": fmt_num(pwr_after, 3),
        "power_rise_w": fmt_num(pwr_rise_w, 4),

        # PCIe
        "pcie_gen": pcie_gen,
        "pcie_width": pcie_width,
        "pcie_efficiency_pct": fmt_num(pcie_eff, 4),
        "pcie_tx_avg_gbs": fmt_num(tx_gbs, 4),
        "pcie_rx_avg_gbs": fmt_num(rx_gbs, 4),
        "pcie_combined_gbs": fmt_num(comb_gbs, 4),
        "pcie_verdict": pcie_verdict,

        # Memory
        "prefetch_gbs": fmt_num(prefetch_gbs, 4),
        "mem_verdict": mem_verdict,
        "naive_skip_reason": naive_skip_reason,

        # Baseline
        "theoretical_fp32_gflops": fmt_num(theo_fp32, 4),
        "realistic_baseline_gflops": fmt_num(realistic_base, 4),
        "healthy_min_gflops": fmt_num(healthy_min, 4),
        "healthy_max_gflops": fmt_num(healthy_max, 4),

        # Identity
        "gpu_name": gpu_name,
        "uuid": uuid,
    }
    return row


# -----------------------------
# Main
# -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Enrich PascalVal training CSV by extracting telemetry from run.json files.")
    ap.add_argument("--root", required=True, help="Root directory containing <RID>/run.json (e.g., results/json)")
    ap.add_argument("--out", required=True, help="Output enriched CSV path")
    ap.add_argument("--rid-list", default="", help="Optional RID list file (one RID per line)")
    ap.add_argument("--strict", action="store_true",
                    help="If set, error if any RID in --rid-list is missing; otherwise skip missing.")
    args = ap.parse_args()

    rid_list: Optional[List[str]] = None
    if args.rid_list.strip():
        rid_list = read_rid_list(args.rid_list.strip())

    rows_out: List[Dict[str, str]] = []
    missing: List[str] = []

    items = list(iter_runjson_paths(args.root, rid_list))
    # stable sort by RID
    items.sort(key=lambda t: t[0])

    for rid, path in items:
        try:
            with open(path, "r") as f:
                doc = json.load(f)
            row = extract_row(rid, doc)
            rows_out.append(row)
        except FileNotFoundError:
            missing.append(rid)
        except json.JSONDecodeError:
            # treat as missing/corrupt
            missing.append(rid)

    if rid_list is not None and args.strict:
        # if strict mode, verify all rids were processed
        processed = {r["rid"] for r in rows_out}
        for rid in rid_list:
            if rid not in processed:
                missing.append(rid)
        missing = sorted(set(missing))
        if missing:
            raise SystemExit(f"[enrich_training_csv] strict mode: missing/corrupt run.json for {len(missing)} RIDs (first 10): {missing[:10]}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # fixed header order
    fieldnames = [
        "rid", "final",
        "engine", "mode", "n", "reps", "gflops", "time_ms", "theoretical_ratio",

        "clock_before_mhz", "clock_during_mhz", "clock_after_mhz", "clock_drop_pct",
        "temp_before_c", "temp_during_c", "temp_after_c", "temp_rise_c",
        "power_before_w", "power_during_w", "power_after_w", "power_rise_w",

        "pcie_gen", "pcie_width", "pcie_efficiency_pct",
        "pcie_tx_avg_gbs", "pcie_rx_avg_gbs", "pcie_combined_gbs", "pcie_verdict",

        "prefetch_gbs", "mem_verdict", "naive_skip_reason",

        "theoretical_fp32_gflops", "realistic_baseline_gflops", "healthy_min_gflops", "healthy_max_gflops",

        "gpu_name", "uuid",
    ]

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    print(f"[enrich_training_csv] {datetime.now().astimezone().isoformat()}")
    print(f"[enrich_training_csv] root={args.root}")
    if rid_list is not None:
        print(f"[enrich_training_csv] rid_list={args.rid_list} (requested {len(rid_list)})")
    print(f"[enrich_training_csv] wrote={args.out} rows={len(rows_out)} missing={len(set(missing))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
