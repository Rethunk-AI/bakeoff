"""Hardware context fields and the schema they land in (bakeoff#38)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import patch

from bench import hardware

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = (ROOT / "schema" / "schema.sql").read_text()


def _table(name: str) -> str:
    m = re.search(rf"CREATE TABLE {name} \((.*?)\n\);", SCHEMA, re.DOTALL)
    assert m, f"no table {name}"
    return m.group(1)


def test_split_pci_id_reads_device_high_and_vendor_low():
    assert hardware.split_pci_id("0x268410DE") == ("0x10de", "0x2684")
    assert hardware.split_pci_id("0x2684") is None
    assert hardware.split_pci_id("N/A") is None


def test_pcie_description_matches_the_seeded_interface_types():
    assert hardware.pcie_description("4", "16") == "PCIe 4.0 x16"
    assert hardware.pcie_description("[N/A]", "16") is None
    seeded = {
        r["description"]
        for r in json.loads((ROOT / "schema/seeds/interface_types.json").read_text())["rows"]
    }
    assert {"PCIe 4.0 x16", "PCIe 3.0 x4", "PCIe 5.0 x8"} <= seeded


def test_nvidia_info_carries_the_new_fields():
    line = (
        "NVIDIA GeForce RTX 4090, 24564, 450.00, 560.35, 0x268410DE, 0x16F410DE, "
        "10501, 2520, 4, 16, 3, 8"
    )
    with patch("bench.hardware._run", return_value=line):
        info = hardware._nvidia_info()
    assert info["gpu_model"] == "NVIDIA GeForce RTX 4090"
    assert info["vram_mb"] == 24564
    assert info["driver_version"] == "560.35"
    assert (info["pci_vendor_id"], info["pci_device_id"]) == ("0x10de", "0x2684")
    assert (info["pci_subsystem_vendor_id"], info["pci_subsystem_device_id"]) == (
        "0x10de",
        "0x16f4",
    )
    assert info["clock_memory_mhz"] == 10501
    assert info["clock_graphics_boost_mhz"] == 2520
    assert info["slot_native_interface"] == "PCIe 4.0 x16"
    assert info["actual_interface"] == "PCIe 3.0 x8"


def test_nvidia_info_tolerates_unsupported_fields():
    line = "GPU, 8192, [N/A], 550.1, [N/A], [N/A], [N/A], [N/A], [N/A], [N/A], [N/A], [N/A]"
    with patch("bench.hardware._run", return_value=line):
        info = hardware._nvidia_info()
    assert info == {"gpu_model": "GPU", "vram_mb": 8192, "driver_version": "550.1"}


def test_context_has_every_new_key_even_without_a_gpu():
    with patch("bench.hardware._run", return_value=""):
        ctx = hardware.collect_hardware_context()
    for key in (
        "pci_vendor_id",
        "pci_device_id",
        "pci_subsystem_vendor_id",
        "pci_subsystem_device_id",
        "clock_memory_mhz",
        "clock_graphics_boost_mhz",
        "slot_native_interface",
        "actual_interface",
    ):
        assert key in ctx and ctx[key] is None


def test_schema_has_every_field_issue_38_lists():
    gpu = _table("gpu_hardware")
    for col in (
        "pci_vendor_id",
        "pci_device_id",
        "pci_subsystem_vendor_id",
        "pci_subsystem_device_id",
        "vram_type_id",
        "gpu_architecture_id",
        "memory_bandwidth_peak_gb_s",
        "tdp_w",
        "tflops_source_id",
    ):
        assert re.search(rf"\b{col}\b", gpu), col
    assert "REFERENCES vram_type" in gpu and "REFERENCES gpu_architecture" in gpu
    for table in ("tflops_source", "vram_type", "gpu_architecture"):
        _table(table)
    link = _table("system_gpu_link")
    assert "slot_native_interface_type_id" in link and "actual_interface_type_id" in link
    assert "PRIMARY KEY (system_hardware_id, slot_index)" in link
    iface = _table("interface_type")
    for col in (
        "bandwidth_peak_gb_s",
        "description",
        "interface_family",
        "lane_transfer_rate",
        "lane_count",
    ):
        assert col in iface, col
    assert "transfer_rate" not in iface.replace("lane_transfer_rate", "")
    sysh = _table("system_hardware")
    assert "system_id          UUID NOT NULL UNIQUE" in sysh and "ram_total_gb" in sysh
    syss = _table("system_software")
    for col in (
        "os",
        "kernel_version",
        "python_version",
        "gpu_driver_version",
        "cuda_version",
        "rocm_version",
        "runner_version",
    ):
        assert re.search(rf"\b{col}\b", syss), col
    rhm = _table("run_hardware_metrics")
    assert "fk_run_hardware_metrics_system_gpu_link" in rhm and "system_software_id" in rhm


def test_interface_seed_bandwidth_follows_the_pcie_formula_and_matches_the_sql():
    rows = json.loads((ROOT / "schema/seeds/interface_types.json").read_text())["rows"]
    pcie = [r for r in rows if r["interface_family"] == "PCIe"]
    assert len(pcie) == 20
    for r in pcie:
        assert r["bandwidth_peak_gb_s"] == r["lane_transfer_rate"] * r["lane_count"] * 2 / 8
    families = {r["description"]: r["interface_family"] for r in rows}
    for name in (
        "SXM2",
        "SXM4",
        "SXM5",
        "NVLink 2.0",
        "NVLink 3.0",
        "NVLink 4.0",
        "Thunderbolt 3",
        "Thunderbolt 4",
        "OCuLink 2.0",
    ):
        assert name in families
    for r in rows:
        assert f"'{r['description']}'" in SCHEMA
