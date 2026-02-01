#!/usr/bin/env python3
"""
PCIe validation using professional pcie_diag tool
"""
import subprocess
import re
from pathlib import Path

def run_all(gpu_caps, **kwargs):
    """Run PCIe diagnostic with the real pcie_diag tool"""
    
    binary = Path(__file__).parent.parent.parent / "public_release" / "pcie_diag"
    
    print()
    
    # Run tool and capture output
    result = subprocess.run(
        ["sudo", str(binary), "1024", "--all-gpus"],
        capture_output=True,
        text=True
    )
    
    # Debug: Save output to file
    with open('/tmp/pcie_debug_output.txt', 'w') as f:
        f.write(f"RETURNCODE: {result.returncode}\n")
        f.write(f"STDOUT:\n{result.stdout}\n")
        f.write(f"STDERR:\n{result.stderr}\n")
    
    # Print output for user
    print(result.stdout)
    
    # Parse bandwidth values
    tx_match = re.search(r'TX avg:\s+([\d.]+)\s+GB/s', result.stdout)
    rx_match = re.search(r'RX avg:\s+([\d.]+)\s+GB/s', result.stdout)
    combined_match = re.search(r'Combined:\s+([\d.]+)\s+GB/s', result.stdout)
    efficiency_match = re.search(r'Efficiency:\s+([\d.]+)%', result.stdout)
    
    tx_avg = float(tx_match.group(1)) if tx_match else 0
    rx_avg = float(rx_match.group(1)) if rx_match else 0
    combined = float(combined_match.group(1)) if combined_match else 0
    efficiency = float(efficiency_match.group(1)) if efficiency_match else 0
    
    verdict = "OK" if result.returncode == 0 else "FAILED"
    
    return {
        "module": "pcie",
        "verdict": verdict,
        "tx_avg_gbs": tx_avg,
        "rx_avg_gbs": rx_avg,
        "combined_gbs": combined,
        "efficiency_pct": efficiency,
    }
