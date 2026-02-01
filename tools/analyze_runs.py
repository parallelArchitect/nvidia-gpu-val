#!/usr/bin/env python3
"""
Standalone analyzer for PascalVal results.

Contract:
  - Reads only: results/json/*/run.json
  - Writes only: sandbox_new/data/log_analyzer_report.json (or --out)
  - Never imports pascalval, never imports run_validation
  - Never modifies any files under results/

Usage:
  python3 tools/analyze_runs.py --take 300 --engine all
"""

from __future__ import annotations
import argparse
import glob
import json
import os
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

ALLOWED_VERDICTS = {"OK", "FAILED"}

@dataclass
class RunRow:
    rid: str
    path: str
    pcie_verdict: str
    mem_verdict: str
    comp_verdict: str
    final: str
    engine: str
    gflops: Optional[float]
    um_prefetch_gbs: Optional[float]
    clock_during_mhz: Optional[float]
    power_during_w: Optional[float]
    temp_during_c: Optional[float]

def rid_from_path(p: str) -> str:
    # results/json/<RID>/run.json
    return os.path.basename(os.path.dirname(p))

def get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def norm_verdict(v: Any) -> str:
    return v if isinstance(v, str) and v in ALLOWED_VERDICTS else "FAILED"

def to_float(x: Any) -> Optional[float]:
    return float(x) if isinstance(x, (int, float)) else None

def read_run(path: str) -> Optional[RunRow]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except Exception:
        return None

    rid = rid_from_path(path)
    results = doc.get("results") if isinstance(doc, dict) else None
    if not isinstance(results, dict):
        return RunRow(rid, path, "FAILED", "FAILED", "FAILED", "FAILED", "MISSING",
                      None, None, None, None, None)

    pcie_v = norm_verdict(get(results, "pcie", "verdict"))
    mem_v  = norm_verdict(get(results, "memory", "verdict"))
    comp_v = norm_verdict(get(results, "compute", "verdict"))
    final  = norm_verdict(results.get("final"))

    engine = get(results, "compute", "engine", default="MISSING")
    if not isinstance(engine, str) or not engine:
        engine = "MISSING"

    gflops = to_float(get(results, "compute", "gflops"))

    # Optional fields (only if present in your schema)
    um_prefetch_gbs = to_float(get(results, "memory", "prefetch_gbs"))
    clock_during_mhz = to_float(get(results, "telemetry", "during_test", "clock_mhz"))
    power_during_w   = to_float(get(results, "telemetry", "during_test", "power_w"))
    temp_during_c    = to_float(get(results, "telemetry", "during_test", "temp_c"))

    return RunRow(
        rid=rid,
        path=path,
        pcie_verdict=pcie_v,
        mem_verdict=mem_v,
        comp_verdict=comp_v,
        final=final,
        engine=engine,
        gflops=gflops,
        um_prefetch_gbs=um_prefetch_gbs,
        clock_during_mhz=clock_during_mhz,
        power_during_w=power_during_w,
        temp_during_c=temp_during_c,
    )

def median(xs: List[float]) -> float:
    return float(statistics.median(xs))

def mad(xs: List[float]) -> float:
    m = median(xs)
    return float(statistics.median([abs(x - m) for x in xs]))

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", type=int, default=300, help="Most-recent run.json files to scan")
    ap.add_argument("--engine", choices=["all", "custom", "cublas"], default="all")
    ap.add_argument("--min-valid", type=int, default=10)
    ap.add_argument("--out", default="sandbox_new/data/log_analyzer_report.json")
    args = ap.parse_args()

    paths = sorted(glob.glob("results/json/*/run.json"), reverse=True)[:max(0, args.take)]
    rows: List[RunRow] = [r for p in paths for r in [read_run(p)] if r is not None]

    buckets = {
        "SCANNED": len(rows),
        "PCIE_FAILED": 0,
        "MEMORY_FAILED": 0,
        "COMPUTE_FAILED": 0,
        "VALID_CUSTOM": 0,
        "VALID_CUBLAS": 0,
        "VALID_OTHER_ENGINE": 0,
    }

    samples: Dict[str, List[float]] = {"custom": [], "cublas": []}

    for r in rows:
        if r.pcie_verdict != "OK":
            buckets["PCIE_FAILED"] += 1
            continue
        if r.mem_verdict != "OK":
            buckets["MEMORY_FAILED"] += 1
            continue
        if r.comp_verdict != "OK":
            buckets["COMPUTE_FAILED"] += 1
            continue

        if r.engine in ("custom", "cublas") and r.gflops is not None:
            if r.engine == "custom":
                buckets["VALID_CUSTOM"] += 1
            else:
                buckets["VALID_CUBLAS"] += 1
            samples[r.engine].append(r.gflops)
        else:
            buckets["VALID_OTHER_ENGINE"] += 1

    def stats(engine: str) -> Dict[str, Any]:
        xs = samples.get(engine, [])
        n = len(xs)
        if n < args.min_valid:
            return {"engine": engine, "n_valid": n, "state": "INSUFFICIENT_DATA", "conf": 0.4}
        m = median(xs)
        d = mad(xs)
        return {"engine": engine, "n_valid": n, "state": "OK", "median_gflops": m, "mad_gflops": d, "conf": 0.85}

    engines = ["custom", "cublas"] if args.engine == "all" else [args.engine]

    report = {
        "buckets": buckets,
        "stats": {e: stats(e) for e in engines},
        "note": "Standalone analyzer: reads results/json, writes only to sandbox_new/data.",
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[analyzer] wrote: {args.out}")
    print(f"[schema] scanned={buckets['SCANNED']}")
    for k in ("PCIE_FAILED","MEMORY_FAILED","COMPUTE_FAILED","VALID_CUSTOM","VALID_CUBLAS","VALID_OTHER_ENGINE"):
        print(f"[bucket] {k}={buckets[k]}")
    for e in engines:
        s = report["stats"][e]
        if s["state"] != "OK":
            print(f"[{e}] n_valid={s['n_valid']} state=INSUFFICIENT_DATA conf={s['conf']}")
        else:
            print(f"[{e}] n_valid={s['n_valid']} median={s['median_gflops']:.2f} mad={s['mad_gflops']:.2f} conf={s['conf']}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
