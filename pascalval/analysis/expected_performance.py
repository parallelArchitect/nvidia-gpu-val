"""
Runtime expected-performance check (informational).

Loads trained coefficients (expected_performance_report.json) and estimates expected GFLOPS
from telemetry, then computes residuals + robust z-score.

This module NEVER fails a run. It only annotates results.
"""

import json
import math
from typing import Dict, Optional

FEATURES = [
    "clock_during_mhz",
    "temp_during_c",
    "power_during_w",
    "pcie_efficiency_pct",
    "prefetch_gbs",
]


def _to_float(x):
    try:
        f = float(x)
        return f if math.isfinite(f) else None
    except Exception:
        return None


def load_model(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def predict_expected_gflops(model: Dict, telemetry: Dict) -> Optional[float]:
    coeffs = model.get("coefficients", {})
    bias = coeffs.get("bias", None)
    if bias is None:
        return None

    total = float(bias)
    for k in FEATURES:
        v = _to_float(telemetry.get(k))
        c = coeffs.get(k, None)
        if v is None or c is None:
            return None
        total += float(c) * v
    return total


def classify_residual(z: Optional[float]) -> str:
    if z is None:
        return "UNKNOWN"
    if z <= -6.0:
        return "UNEXPLAINED_LOW"
    if z <= -3.0:
        return "LOW"
    return "NORMAL"


def run_expected_performance(model_path: str,
                             telemetry: Dict,
                             observed_gflops: Optional[float]) -> Dict:
    try:
        model = load_model(model_path)
    except Exception as e:
        return {"available": False, "reason": f"model_load_failed: {e}"}

    exp = predict_expected_gflops(model, telemetry)
    obs = _to_float(observed_gflops)

    if exp is None or obs is None:
        return {"available": False, "reason": "missing_telemetry_or_observed"}

    res = obs - exp

    rstats = model.get("residuals", {})
    r_med = _to_float(rstats.get("median"))
    r_mad = _to_float(rstats.get("mad"))

    z = None
    if r_mad is not None and r_mad > 0 and r_med is not None:
        z = (res - r_med) / (1.4826 * r_mad)

    return {
        "available": True,
        "expected_gflops": round(exp, 3),
        "observed_gflops": round(obs, 3),
        "residual_gflops": round(res, 3),
        "residual_z": round(z, 3) if z is not None else None,
        "classification": classify_residual(z),
        "note": "Informational expected-performance check (non-fatal)"
    }
