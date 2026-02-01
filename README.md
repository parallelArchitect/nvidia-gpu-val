 
 README
 
# PascalVal — Linux NVIDIA GPU Health & Drift Validator

PascalVal is a **Linux-based GPU validation and drift-evaluation tool** for **NVIDIA GPUs**.

It is designed to answer one operational question:

> **Is this GPU and platform operating within expected behavior, or has it drifted?**

PascalVal validates **transport**, **memory residency behavior**, and **compute stability**, producing structured artifacts that can be accumulated and evaluated over time.

---

## What PascalVal Validates

Each execution follows a **gated validation pipeline**:

### 1) PCIe Transport

PascalVal performs a **bounded PCIe transport validation**:

* Verifies negotiated link generation and width
* Measures host↔device bandwidth
* Detects lane drops, fallback modes, and platform-level transport issues

For deep PCIe diagnostics and bandwidth characterization (standalone toolchain), see:

https://github.com/parallelArchitect/gpu-pcie-diagnostic

---

### 2) Unified Memory (UM) Behavior

PascalVal performs a **bounded Unified Memory validation** focused on **device residency behavior**.

* Measures **device-initiated Unified Memory prefetch throughput**
* Explicitly **does not** run naive bulk bandwidth tests
* Avoids misleading results caused by pageable copies or host-side effects

The UM module answers a narrow validation question:

> *Is Unified Memory resident on the device and servicing accesses through the expected fast path?*

For **deep Unified Memory analysis** (migration paths, fault behavior, oversubscription), see:

https://github.com/parallelArchitect/pascal-um-benchmark

---

### 3) FP32 SGEMM Compute Stability

* Executes a fixed-size SGEMM workload representative of scientific and ML compute
* Reports achieved GFLOPS relative to theoretical capability

If a gate fails, downstream stages **do not execute**.  
This prevents misleading compute results when the system is already unhealthy.

Each execution produces a single evidence artifact:

```

results/json/<run_id>/run.json

````

This file is the **source of truth** for all analysis.

---

## Supported GPUs and Compute Paths

### Does PascalVal work on all NVIDIA GPUs?

**Yes.**  
PascalVal runs on **all NVIDIA GPUs supported by the Linux NVIDIA driver**.

On every NVIDIA GPU, PascalVal provides:

* PCIe validation
* Unified Memory validation (where supported)
* Compute validation using **cuBLAS**
* Historical evaluation via the analyzer

This is the **portable baseline path**.

---

### Architecture-Specific Compute Routing

On **Pascal GP104 (SM 6.1)** GPUs (for example GTX 1080 / 1070 / 1060-6GB):

* The validator **automatically selects** a custom SGEMM kernel
* The kernel uses a **fixed, Pascal-tuned execution configuration**
* No user tuning or manual selection is required

On all other NVIDIA GPUs, PascalVal automatically uses **cuBLAS** as the compute validation path.

This routing decision is **policy-driven** and recorded in each `run.json` artifact.

---

## Custom SGEMM Binary Hardening (Linux)

The Pascal GP104 custom SGEMM binary (`sgemm_gp104`) is built as a hardened Linux executable:

* PIE (ASLR)
* Non-executable stack (NX)
* Immediate binding (RELRO / BIND_NOW)

The binary is non-networked, dynamically linked, and performs compute only.  
It does not modify system state.

---

## cuBLAS vs Custom SGEMM — Engineering Characteristics

Repeated validation under controlled conditions shows expected differences:

### cuBLAS (all NVIDIA GPUs)

* General-purpose, adaptive library
* Runtime kernel selection and scheduling
* Sensitive to boost state, power management, and background activity

**Role:** Portable reference and sanity check.

### Custom SGEMM (Pascal GP104 / SM 6.1 only)

* Fixed kernel configuration and launch geometry
* Minimal runtime decision-making
* Lower variance and higher determinism under identical conditions

**Role:** Preferred signal for **long-term health evaluation** on supported Pascal GPUs.

> cuBLAS answers: *“Is the system broadly functional?”*  
> Custom SGEMM answers: *“Is this Pascal GPU operating within its expected deterministic compute envelope?”*

Both are valid.  
They are **not equivalent signals**.

---

## Installation

PascalVal is a Linux-based tool for NVIDIA GPUs.

**Requirements**
- Linux
- NVIDIA GPU with a working NVIDIA driver (NVML available)
- Python 3.9+

```bash
git clone https://github.com/parallelArchitect/pascal-gpu-val.git
cd pascal-gpu-val
pip install -r requirements.txt
````

---

## Usage

### Run a validation (default)

```bash
python3 run_validation.py
```

This executes the gated pipeline and writes one immutable evidence artifact to:

```
results/json/<run_id>/run.json
```

---

### Validate all GPUs in the system

```bash
python3 run_validation.py --all-gpus
```

Runs one validation per GPU and produces one artifact per GPU.

---

### Analyze historical behavior

```bash
python3 tools/analyze_runs.py --engine all --out results/analysis/analyzer_report.json
```

The analyzer:

* Reads existing `run.json` artifacts (read-only)
* Builds robust baselines (median + MAD)
* Reports whether deviations are isolated or persistent over time

---

## Operational Model

PascalVal is intended to be executed **repeatedly over time** as part of normal system operation.

* Each execution appends one immutable artifact under `results/json/`
* Artifacts are **append-only**
* No single run is treated as authoritative

Behavior is evaluated across **multiple runs** using the analyzer.

---

## Analyzer (Historical Evaluation)

### Minimum Evidence Requirement

Meaningful historical analysis requires **multiple runs**.

* Fewer than **12 valid runs** → results are reported as *insufficient data*
* **12+ valid runs** → baseline statistics (median, MAD) become stable
* Larger histories improve confidence and drift sensitivity

The analyzer explicitly reports when available evidence is insufficient to support conclusions.

The analyzer operates **read-only** on existing `run.json` artifacts.

It evaluates recent history to determine whether observed deviations are:

* **Isolated** (expected variance), or
* **Persistent over time** (behavioral drift)

Baselines are derived using **robust statistics (median and MAD)**.

Derived reports are written to:

```
results/analysis/analyzer_report.json
```

---

## Results Layout (Official)

```
results/
  json/<run_id>/run.json          # primary evidence artifact (append-only)
  analysis/analyzer_report.json   # derived report (read-only)
```

---

## Architecture Overview

See the full system diagram and file-level boundaries in
[`docs/architecture/ARCHITECTURE.md`](docs/architecture/ARCHITECTURE.md).

PascalVal is intentionally split into two roles:

### Right Brain — Measurement
- PCIe
- Unified Memory
- Compute

### Left Brain — Decision & Policy
- Gate ordering
- Evaluation
- Verdict generation
- Artifact writing

---

## What PascalVal Is (and Is Not)


**PascalVal is:**

* A Linux NVIDIA GPU health validator
* A drift-evaluation tool
* A pre-compute sanity check

**PascalVal is not:**

* A peak benchmark
* A tuning framework
* An application profiler

---

## Analysis Tools (Offline)

The `tools/` directory contains optional, offline utilities that analyze historical
`run.json` artifacts produced by PascalVal. These tools operate in **read-only mode**
and never influence validation behavior or results.

Typical uses include:

* Aggregating multiple validation runs
* Evaluating variance and persistent drift over time
* Producing summary reports for long-term GPU health monitoring

---

## License

MIT License

---

**Author:**
**Joe McLaren** — *Human–AI collaborative engineering*

```

---



