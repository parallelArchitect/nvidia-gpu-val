#!/usr/bin/env python3
"""
Author: Joe McLaren (Human–AI collaborative engineering)
Project: pascal-gpu-val
File: pascalval/modules/cublas_sgemm.py

Description:
  cuBLAS SGEMM fallback engine for non-Pascal (or when custom engine isn't applicable).
  Contract: numbers only (no OK/FAILED, no baselines, no intelligence).
  Produces:
    - engine="cublas"
    - n, reps, gflops, time_ms

Implementation:
  Uses an external binary if present:
    1) ./cublas_sgemm_bench (repo root)
    2) public_release/pascalval_cublas_sgemm_bench
  Expected --quiet output: "N,gflops,ms"
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Any, Dict, Optional

@dataclass
class CublasSgemmResult:
    engine: str
    n: int
    reps: int
    gflops: float
    time_ms: float

def _find_binary() -> Optional[Path]:
    # Prefer repo-local bench if you compiled it
    p1 = Path.cwd() / "cublas_sgemm_bench"
    if p1.exists():
        return p1
    p2 = Path.cwd() / "public_release" / "pascalval_cublas_sgemm_bench"
    if p2.exists():
        return p2
    return None

def run_all(caps: Dict[str, Any], *, n: int = 4096, reps: int = 20) -> Dict[str, Any]:
    binary = _find_binary()
    if not binary:
        return {
            "module": "sgemm",
            "engine": "cublas",
            "error": "MISSING_CUBLAS_BINARY",
            "n": n,
            "reps": reps,
        }

    try:
        r = subprocess.run(
            [str(binary), "--n", str(n), "--reps", str(reps), "--quiet"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = (r.stdout or "").strip()
        # expected: N,gflops,ms
        parts = out.split(",")
        if len(parts) != 3:
            raise ValueError(f"Unexpected --quiet output: {out!r}")

        N = int(parts[0].strip())
        gflops = float(parts[1].strip())
        ms = float(parts[2].strip())

        return {
            "module": "sgemm",
            "engine": "cublas",
            "n": N,
            "reps": reps,
            "gflops": gflops,
            "time_ms": ms,
        }

    except Exception as e:
        return {
            "module": "sgemm",
            "engine": "cublas",
            "error": f"ERROR: {e}",
            "n": n,
            "reps": reps,
        }
