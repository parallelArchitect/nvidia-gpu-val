
import ast
from pathlib import Path
import pytest
from pascalval.pascal_kb import PascalKnowledgeBase

def test_gp104():
    c = PascalKnowledgeBase().interpret_gpu(dict(
        pci_device_id="0x1b80", compute_capability_major=6,
        compute_capability_minor=1, multiprocessor_count=20,
        memory_free_mb=7000))
    assert c["silicon_id"] == "GP104"
    assert c["cuda_cores_total"] == 2560

@pytest.mark.parametrize("major,minor", [(7,5),(8,6),(8,9),(12,0)])
def test_rtx_capabilities(major, minor):
    c = PascalKnowledgeBase().interpret_gpu(dict(
        pci_device_id="0xffff", compute_capability_major=major,
        compute_capability_minor=minor, multiprocessor_count=80,
        memory_free_mb=16000))
    assert c["compute_capability"] == f"{major}.{minor}"
    assert c["cuda_cores_total"] is None
    assert c["safe_test_max_mb"] == 15200

@pytest.mark.parametrize("silicon,cc,expected", [
    ("GP104","6.1","custom"),
    ("UNKNOWN","7.5","cublas"),
    ("UNKNOWN","8.6","cublas"),
    ("UNKNOWN","8.9","cublas"),
    ("UNKNOWN","12.0","cublas"),
])
def test_routing(tmp_path, monkeypatch, silicon, cc, expected):
    monkeypatch.chdir(tmp_path)
    binary = tmp_path / "public_release/pascalval_sgemm_public"
    binary.parent.mkdir()
    binary.touch()
    source = ast.parse((Path(__file__).resolve().parents[1] /
                        "run_validation.py").read_text())
    fn = next(n for n in source.body
              if isinstance(n, ast.FunctionDef)
              and n.name == "_compute_route")
    scope = {"Path": Path, "Dict": dict, "Any": object}
    exec(compile(ast.Module(body=[fn], type_ignores=[]),
                 "<routing>", "exec"), scope)
    result = scope["_compute_route"](
        {"silicon_id": silicon, "compute_capability": cc}, "auto")
    assert result["selected"] == expected
