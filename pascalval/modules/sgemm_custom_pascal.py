ENABLE_UI = False

#!/usr/bin/env python3
"""
Author: Emile (Human-AI collaborative engineering)
Project: pascal-gpu-val
File: pascalval/modules/sgemm.py

SGEMM Compute Validation with Intelligent Boost Management
"""

import subprocess
import re
import time
from pathlib import Path
from typing import Dict
import pynvml

try:
    from rich.console import Console
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from . import boost


def run_all(gpu_caps: Dict, initial_state=None, **kwargs) -> Dict:
    """
    SGEMM validation with intelligent boost management
    
    Flow:
    1. Check current GPU state (likely idle)
    2. Warmup to force boost state if needed
    3. Monitor during warmup
    4. Run SGEMM test in boosted state
    5. Report with state context
    """
    
    console = Console() if RICH_AVAILABLE else None
    
    if console:
        if ENABLE_UI: console.print("\n[bold cyan]═══════════════════════════════════════════════[/bold cyan]")
        if ENABLE_UI: console.print("[bold cyan]SGEMM Compute Validation[/bold cyan]")
        if ENABLE_UI: console.print("[bold cyan]═══════════════════════════════════════════════[/bold cyan]\n")
    else:
        if ENABLE_UI: print("\n" + "="*60)
        if ENABLE_UI: print("SGEMM COMPUTE VALIDATION")
        if ENABLE_UI: print("="*60 + "\n")
    
    gpu_name = gpu_caps['name']
    silicon_id = gpu_caps['silicon_id']
    baseline = gpu_caps.get("performance_baseline", {})
    baseline_gflops = baseline.get("realistic_baseline_gflops", 7000)
    
    # Phase 1: Check initial state
    if initial_state is None:
        initial_state = _get_gpu_state(0)
    
    if console:
        if ENABLE_UI: console.print(f"GPU: [green]{gpu_name}[/green] ([cyan]{silicon_id}[/cyan])")
        if ENABLE_UI: console.print(f"Initial state: {initial_state['clocks_mhz']} MHz, {initial_state['temp_c']}°C")
    
    # Phase 2: Warmup if needed
    warmup_result = None
    if initial_state['clocks_mhz'] < 1500:
        if console:
            if ENABLE_UI: console.print("[yellow]GPU in idle state - warming up to boost...[/yellow]")
        
        warmup_result = _warmup_gpu(console, duration_s=30)
        
        if console:
            final_clocks = warmup_result['final_clocks_mhz']
            if final_clocks >= 1500:
                if ENABLE_UI: console.print(f"[green]✓ GPU boosted to {final_clocks} MHz[/green]\n")
            else:
                if ENABLE_UI: console.print(f"[yellow]⚠ GPU at {final_clocks} MHz (may not be fully boosted)[/yellow]\n")
    else:
        if console:
            if ENABLE_UI: console.print("[green]✓ GPU already in boost state[/green]\n")
    
    # Phase 3: Run SGEMM with monitoring
    if console:
        pass
    
    # Run test
    test_result = _run_sgemm_test(4096, 5, console)
    
    # Sample final state
    final_state = _get_gpu_state(0)
    
    # Phase 4: Display results with context
    _display_results(
        test_result,
        baseline_gflops,
        initial_state,
        final_state,
        warmup_result,
        console
    )
    
    # Determine verdict
    gflops = test_result['gflops']
    
    if gflops == 0:
        verdict = "FAILED"
    elif gflops >= (baseline_gflops * 0.90):
        verdict = "HEALTHY"
    elif gflops >= (baseline_gflops * 0.75):
        verdict = "DEGRADED"
    else:
        verdict = "FAILED"
    
    return {
        "module": "sgemm",
        "result": test_result,
        "baseline_gflops": baseline_gflops,
        "initial_state": initial_state,
        "final_state": final_state,
        "warmup": warmup_result,
        "verdict": verdict,
    }


def _warmup_gpu(console, duration_s=30) -> Dict:
    """
    Warmup GPU to boost state using lightweight compute
    
    Uses boost.py to monitor state transitions
    """
    
    binary = Path(__file__).parent.parent.parent / "public_release" / "pascalval_sgemm_public"
    
    start_time = time.time()
    iterations = 0
    
    # Run boost monitoring in background
    boost_data = boost.run_all(
        duration_s=duration_s,
        interval_s=0.5,
        progress=False,
    )
    
    # Lightweight warmup loop - keep NVML open during loop
    pynvml.nvmlInit()
    peak_state = None
    
    while (time.time() - start_time) < duration_s:
        # Small SGEMM to generate heat/work
        subprocess.run(
            [str(binary), "2048", "1"],
            capture_output=True,
            timeout=5
        )
        iterations += 1
        
        # Check if boosted and capture peak state
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        try:
            clocks = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
            
            state = {"clocks_mhz": clocks, "temp_c": temp, "power_w": power}
            
            if peak_state is None or clocks > peak_state['clocks_mhz']:
                peak_state = state
            
            if clocks >= 1500:
                # Boosted! Continue for a bit more to stabilize
                if (time.time() - start_time) >= 10:
                    break
        except:
            pass
        
        time.sleep(0.2)
    
    pynvml.nvmlShutdown()
    
    # If no peak captured, get current state
    if peak_state is None:
        peak_state = _get_gpu_state(0)
    
    return {
        "duration_s": time.time() - start_time,
        "iterations": iterations,
        "final_clocks_mhz": peak_state['clocks_mhz'],
        "final_temp_c": peak_state['temp_c'],
        "final_power_w": peak_state.get('power_w', 0),
        "boost_data": boost_data,
    }


def _get_gpu_state(gpu_index=0) -> Dict:
    """Get current GPU state via NVML"""
    
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    
    try:
        clocks = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
    except:
        clocks = 0
    
    try:
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    except:
        temp = 0
    
    try:
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW to W
    except:
        power = 0
    
    pynvml.nvmlShutdown()
    
    return {
        "clocks_mhz": clocks,
        "temp_c": temp,
        "power_w": power,
    }


def _run_sgemm_test(size: int, reps: int, console) -> Dict:
    """Run SGEMM test (GPU should already be warmed up)"""
    
    binary = Path(__file__).parent.parent.parent / "public_release" / "pascalval_sgemm_public"
    
    if not binary.exists():
        return {
            "size": size,
            "gflops": 0,
            "time_ms": 0,
            "error": "Binary not found"
        }
    
    # Run test
    result = subprocess.run(
        [str(binary), str(size), str(reps)],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    # Parse output: "N=4096  time=19.200 ms  GFLOPS=7158.1"
    output = result.stdout
    
    gflops_match = re.search(r'GFLOPS=([\d.]+)', output)
    time_match = re.search(r'time=([\d.]+)\s*ms', output)
    
    gflops = float(gflops_match.group(1)) if gflops_match else 0
    time_ms = float(time_match.group(1)) if time_match else 0
    
    return {
        "size": size,
        "reps": reps,
        "gflops": gflops,
        "time_ms": time_ms,
        "raw_output": output
    }


def _display_results(test_result, baseline_gflops, initial_state, final_state, warmup_result, console):
    """Display results with GPU state context"""
    
    if console and RICH_AVAILABLE:
        # Performance table
        table = Table(title="SGEMM Performance Results", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")
        
        table.add_row("Matrix Size", f"{test_result['size']}×{test_result['size']}")
        table.add_row("Performance", f"{test_result['gflops']:.1f} GFLOPS")
        table.add_row("Time", f"{test_result['time_ms']:.2f} ms")
        
        # Determine status
        if test_result["gflops"] > 7000:
            status = "[green]HEALTHY ✓[/green]"
        elif test_result["gflops"] > 5000:
            status = "[yellow]DEGRADED ⚠[/yellow]"
        else:
            status = "[red]FAILED ✗[/red]"
        table.add_row("Status", status)
        
        if ENABLE_UI: console.print()
        if ENABLE_UI: console.print(table)
        
        # GPU state context
        state_table = Table(title="GPU State During Test", show_header=True, header_style="bold yellow")
        state_table.add_column("Phase", style="cyan")
        state_table.add_column("Clocks", style="yellow", justify="right")
        state_table.add_column("Temp", style="yellow", justify="right")
        state_table.add_column("Power", style="yellow", justify="right")
        
        state_table.add_row(
            "Before",
            f"{initial_state['clocks_mhz']} MHz",
            f"{initial_state['temp_c']}°C",
            f"{initial_state['power_w']:.1f}W"
        )
        
        if warmup_result:
            state_table.add_row(
                "After Warmup",
                f"{warmup_result['final_clocks_mhz']} MHz",
                f"{warmup_result['final_temp_c']}°C",
                f"{warmup_result.get('final_power_w', 0):.1f}W"
            )
        
        state_table.add_row(
            "During Test",
            f"{final_state['clocks_mhz']} MHz",
            f"{final_state['temp_c']}°C",
            f"{final_state['power_w']:.1f}W"
        )
        
        if ENABLE_UI: console.print()
        if ENABLE_UI: console.print(state_table)
        if ENABLE_UI: console.print()
    
    else:
        # Plain text
        if ENABLE_UI: print(f"\nResults:")
        if ENABLE_UI: print(f"  Performance: {test_result['gflops']:.1f} GFLOPS")
        if ENABLE_UI: print(f"  Time: {test_result['time_ms']:.2f} ms")
        if ENABLE_UI: print(f"\nGPU State:")
        if ENABLE_UI: print(f"  Before: {initial_state['clocks_mhz']} MHz, {initial_state['temp_c']}°C")
        if ENABLE_UI: print(f"  During: {final_state['clocks_mhz']} MHz, {final_state['temp_c']}°C")
        if ENABLE_UI: print()
