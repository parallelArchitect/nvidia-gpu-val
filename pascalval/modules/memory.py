#!/usr/bin/env python3
"""
Author: Joe McLaren (Human–AI collaborative engineering)
Project: pascal-gpu-val
File: pascalval/modules/memory.py

Description:
  Execution-only Unified Memory bandwidth module.
  - Runs the compiled public_release/pascalval_memory_test
  - Captures stdout (does not print)
  - Returns a dict only (UI + policy live in orchestrator)

Contract:
  run_all(caps, **kwargs) -> dict

Notes:
  This module intentionally emits no console output. The orchestrator
  renders tables/UI from the returned dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import subprocess


def _repo_root() -> Path:
    # .../pascalval/modules/memory.py -> repo root is 3 parents up
    return Path(__file__).resolve().parents[2]


def _bin_path() -> Path:
    return _repo_root() / "public_release" / "pascalval_memory_test"


def _first_float(text: str) -> Optional[float]:
    """
    Extract the first floating-point number from a string without regex.
    Accepts patterns like: 242.3, 242, 242.3GB/s, etc.
    """
    s = text.strip()
    buf = []
    seen_digit = False
    seen_dot = False

    for ch in s:
        if ch.isdigit():
            buf.append(ch)
            seen_digit = True
            continue
        if ch == "." and not seen_dot:
            # allow leading dot only if we already saw digits? no — accept ".5" too
            buf.append(ch)
            seen_dot = True
            continue

        # stop once we have a plausible number and we hit a non-number char
        if seen_digit and buf:
            break

        # otherwise keep scanning
        if not seen_digit and ch not in "+-":
            continue
        if ch in "+-" and not buf:
            buf.append(ch)

    if not buf:
        return None

    try:
        return float("".join(buf))
    except Exception:
        return None


def _parse_output(stdout: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Supports multiple output styles. We look for:
      - lines containing "Naive UM" / "Prefetch UM"
      - or a table with GB/s values (we pick the largest as prefetch if only one is present)
    """
    naive = None
    prefetch = None

    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    for ln in lines:
        low = ln.lower()
        if "naive" in low and "um" in low and ("gb/s" in low or "gbs" in low):
            v = _first_float(ln)
            if v is not None:
                naive = v
        if "prefetch" in low and "um" in low and ("gb/s" in low or "gbs" in low):
            v = _first_float(ln)
            if v is not None:
                prefetch = v

    # fallback: if prefetch not found but we have one "VRAM Bandwidth" style value
    if prefetch is None:
        candidates = []
        for ln in lines:
            if "gb/s" in ln.lower() or "gbs" in ln.lower():
                v = _first_float(ln)
                if v is not None:
                    candidates.append(v)
        if candidates:
            # choose max as "prefetch-resident" bandwidth (most tools print the fast path)
            prefetch = max(candidates)

    return naive, prefetch


def run_all(gpu_caps: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
    exe = _bin_path()
    if not exe.exists():
        return {"module": "memory", "verdict": "ERROR", "error": "MISSING_BINARY", "binary": str(exe)}

    # run and capture output (silent module)
    try:
        p = subprocess.run(
            [str(exe)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except Exception as e:
        return {"module": "memory", "verdict": "ERROR", "error": "EXEC_FAILED", "detail": str(e), "binary": str(exe)}

    naive, prefetch = _parse_output(p.stdout or "")

    out: Dict[str, Any] = {
        "module": "memory",
        "binary": str(exe),
        "naive_gbs": naive,
        "prefetch_gbs": prefetch,
    }

    # speedup is useful for forensics/trend charts; keep it (no UI)
    if naive and prefetch and naive > 0:
        out["speedup"] = float(prefetch) / float(naive)

    # verdict policy (keep as-is to match your current run.json behavior)
    # If you later want ALL verdicting in orchestrator, set verdict="OK" always and move gating up.
    if prefetch is None:
        out["verdict"] = "ERROR"
        out["error"] = "PARSE_FAILED"
        out["raw_len"] = len(p.stdout or "")
        return out

    # If your orchestrator already computes threshold, you can downgrade this to "OK" unconditionally.
    # For now: OK when we got a sane prefetch measurement.
    out["verdict"] = "OK"
    return out
