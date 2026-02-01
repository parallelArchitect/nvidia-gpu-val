#!/usr/bin/env python3
"""
Author: Joe McLaren
Project: pascal-gpu-val
File: tools/train_expected_performance.py

Description:
  Train an interpretable expected-performance model from enriched PascalVal CSV.
  Predicts expected GFLOPS from telemetry and computes residuals + robust z-scores.

Design goals:
  - Interpretable (no black box)
  - Robust (median/MAD)
  - No single-run panic
  - No external dependencies

Outputs:
  - expected_performance.csv
  - expected_performance_report.json
"""

import csv
import json
import math
import argparse
from statistics import median
from datetime import datetime
from typing import List, Dict, Optional


# -----------------------------
# Utilities
# -----------------------------

def to_float(x):
    try:
        f = float(x)
        return f if math.isfinite(f) else None
    except Exception:
        return None


def med(xs):
    xs = [x for x in xs if x is not None]
    return median(xs) if xs else None


def mad(xs, m=None):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    if m is None:
        m = median(xs)
    return median([abs(x - m) for x in xs])


def robust_z(x, m, mad_val, eps=1e-9):
    return (x - m) / (1.4826 * max(mad_val, eps))


# -----------------------------
# Feature extraction
# -----------------------------

FEATURES = [
    "clock_during_mhz",
    "temp_during_c",
    "power_during_w",
    "pcie_efficiency_pct",
    "prefetch_gbs",
]

TARGET = "gflops"


def extract_features(row: Dict[str, str]) -> Optional[List[float]]:
    feats = []
    for k in FEATURES:
        v = to_float(row.get(k, ""))
        if v is None:
            return None
        feats.append(v)
    return feats


# -----------------------------
# Simple linear regression (closed form, no libs)
# y = b0 + b1*x1 + ...
# -----------------------------

def train_linear(X: List[List[float]], y: List[float]):
    """
    Very small closed-form linear regression using normal equations.
    Assumes X is small and well-conditioned (true here).
    """
    n = len(X)
    m = len(X[0])

    # Add bias column
    Xb = [[1.0] + row for row in X]

    # Compute X^T X and X^T y
    XtX = [[0.0] * (m + 1) for _ in range(m + 1)]
    Xty = [0.0] * (m + 1)

    for i in range(n):
        for j in range(m + 1):
            Xty[j] += Xb[i][j] * y[i]
            for k in range(m + 1):
                XtX[j][k] += Xb[i][j] * Xb[i][k]

    # Solve via Gaussian elimination
    # Augmented matrix
    A = [XtX[i] + [Xty[i]] for i in range(m + 1)]

    for i in range(m + 1):
        # pivot
        pivot = A[i][i]
        if abs(pivot) < 1e-12:
            continue
        for j in range(i, m + 2):
            A[i][j] /= pivot
        for r in range(m + 1):
            if r != i:
                factor = A[r][i]
                for c in range(i, m + 2):
                    A[r][c] -= factor * A[i][c]

    coeffs = [A[i][-1] for i in range(m + 1)]
    return coeffs  # [bias, w1, w2, ...]


def predict(coeffs, feats):
    return coeffs[0] + sum(c * f for c, f in zip(coeffs[1:], feats))


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-report", required=True)
    args = ap.parse_args()

    rows = []
    with open(args.csv) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    X, y, kept = [], [], []
    for row in rows:
        feats = extract_features(row)
        target = to_float(row.get(TARGET, ""))
        if feats is None or target is None:
            continue
        X.append(feats)
        y.append(target)
        kept.append(row)

    coeffs = train_linear(X, y)

    # Predict + residuals
    residuals = []
    out_rows = []

    for row, feats, obs in zip(kept, X, y):
        exp = predict(coeffs, feats)
        res = obs - exp
        residuals.append(res)

        out = dict(row)
        out["expected_gflops"] = f"{exp:.3f}"
        out["residual_gflops"] = f"{res:.3f}"
        out_rows.append(out)

    # Robust stats on residuals
    r_med = med(residuals)
    r_mad = mad(residuals, r_med)

    for out in out_rows:
        res = to_float(out["residual_gflops"])
        z = robust_z(res, r_med, r_mad) if r_mad else 0.0
        out["residual_z"] = f"{z:.3f}"

    # Write CSV
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    # Report
    report = {
        "schema": "pascalval.expected_performance.v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "features": FEATURES,
        "coefficients": {
            "bias": coeffs[0],
            **{f: c for f, c in zip(FEATURES, coeffs[1:])},
        },
        "residuals": {
            "median": r_med,
            "mad": r_mad,
        },
        "interpretation": {
            "note": "Residuals represent unexplained performance after accounting for telemetry.",
            "guideline": "Large negative residual_z indicates underperformance not explained by clock/temp/power/pcie/memory.",
        },
    }

    with open(args.out_report, "w") as f:
        json.dump(report, f, indent=2)

    print(args.out_csv)
    print(args.out_report)


if __name__ == "__main__":
    main()
