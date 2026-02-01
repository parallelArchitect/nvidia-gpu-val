#!/usr/bin/env python3
"""
Author: Joe McLaren (Human–AI collaborative engineering)
Project: pascal-gpu-val
File: pascalval/modules/sgemm_engine.py

Description:
  Execution-only SGEMM engine wrapper.
  Contract:
    - accepts engine="custom"|"cublas"
    - runs it
    - returns numbers only
    - no UI, no OK/FAILED, no baselines, no intelligence

Notes:
  Custom path executes public_release/pascalval_sgemm_public and CAPTURES stdout
  so nothing prints to console.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, Dict


_RE_LINE = re.compile(r"N\s*=\s*(\d+).*?time\s*=\s*([0-9.]+)\s*ms.*?GFLOPS\s*=\s*([0-9.]+)", re.IGNORECASE)

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

def _custom_bin() -> Path:
    return _repo_root() / "public_release" / "pascalval_sgemm_public"

def _run_custom(n: int, reps: int) -> Dict[str, Any]:
    b = _custom_bin()
    if not b.exists():
        return {"module": "sgemm", "engine": "custom", "error": "MISSING_CUSTOM_BINARY", "n": n, "reps": reps}

    # Try a few known argument styles without printing anything.
    candidates = [
        [str(b), str(n), str(reps)],
        [str(b), "--n", str(n), "--reps", str(reps)],
        [str(b)],
    ]

    last_err = None
    for cmd in candidates:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            out = (r.stdout or "").strip()

            # Some builds print one parseable line: "N=4096  time=15.49 ms  GFLOPS=8871.9"
            m = _RE_LINE.search(out)
            if m:
                N = int(m.group(1))
                time_ms = float(m.group(2))
                gflops = float(m.group(3))
                return {"module": "sgemm", "engine": "custom", "n": N, "reps": reps, "gflops": gflops, "time_ms": time_ms}

            # If stdout is exactly a CSV "N,gflops,ms" (some builds do this), parse it too
            parts = [p.strip() for p in out.split(",")]
            if len(parts) == 3 and parts[0].isdigit():
                N = int(parts[0]); gflops = float(parts[1]); time_ms = float(parts[2])
                return {"module": "sgemm", "engine": "custom", "n": N, "reps": reps, "gflops": gflops, "time_ms": time_ms}

            last_err = f"UNPARSEABLE_OUTPUT: {out!r}"

        except Exception as e:
            last_err = f"ERROR: {e}"

    return {"module": "sgemm", "engine": "custom", "error": last_err or "UNKNOWN", "n": n, "reps": reps}

def run_all(caps: Dict[str, Any], *, engine: str = "custom", n: int = 4096, reps: int = 20, **kwargs) -> Dict[str, Any]:
    if engine == "cublas":
        from pascalval.modules.sgemm_custom_pascal import run_all as fixed_run_all
        return cublas_run_all(caps, n=n, reps=reps)

    if engine != "custom":
        return {"module": "sgemm", "engine": engine, "error": "UNKNOWN_ENGINE", "n": n, "reps": reps}

    return _run_custom(n=n, reps=reps)
