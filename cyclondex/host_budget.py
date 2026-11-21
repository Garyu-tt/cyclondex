"""Read-only Linux resource preflight for ALT-to-CYCLON checkpoints.

A memory-fit check is NEVER proof of viable inference latency, CPU/GPU
compatibility, quantization correctness, or budgeted training.
"""
import math
from pathlib import Path
import subprocess

GIB = 1024 ** 3
GPU_MXFP4_MIN_CC = (7, 5)


def meminfo(data):
    values = {}
    for line in data.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        columns = value.strip().split()
        if (columns and columns[0].isdigit() and
                (len(columns) == 1 or columns[1] == "kB")):
            values[key] = int(columns[0]) * (1024 if len(columns) > 1 else 1)
    if "MemTotal" not in values or "MemAvailable" not in values:
        raise ValueError("missing Linux memory fields")
    return {"total": values["MemTotal"], "available": values["MemAvailable"]}


def parse_gpu(text):
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        chunks = [part.strip() for part in line.split(",")]
        if len(chunks) != 3:
            continue
        name, mem, capability = chunks
        if (not mem.isdigit() or not name or
                not all(s.isdigit() for s in capability.split(".")) or
                len(capability.split(".")) != 2):
            continue
        cc = tuple(int(s) for s in capability.split("."))
        out.append({"model": name[:100], "memory_mib": int(mem),
                    "compute_capability": capability,
                    "hf_native_mxfp4_supported_by_cc":
                        cc >= GPU_MXFP4_MIN_CC})
    return out


def snapshot():
    ram = meminfo(Path("/proc/meminfo").read_text(encoding="utf-8"))
    try:
        run = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False, timeout=4,
        )
        if run.returncode == 0:
            gpus = parse_gpu(run.stdout)
            gpu_status = "reported_by_nvidia_smi"
        else:
            gpus, gpu_status = [], "driver_or_device_unavailable"
    except (OSError, subprocess.TimeoutExpired):
        gpus, gpu_status = [], "nvidia_smi_not_available"
    return {"ram_bytes": ram, "gpu_status": gpu_status,
            "gpus": gpus, "gpu_vram_is_not_automatically_pooled": True}


def check_candidate(snapshot, estimated_weights_bytes, *,
                    independent_sessions=2, kv_context_gib=1,
                    reserve_gib=8):
    if (type(independent_sessions) is not int or not
            1 <= independent_sessions <= 64 or
            type(estimated_weights_bytes) is not int or
            estimated_weights_bytes <= 0 or
            not math.isfinite(kv_context_gib) or kv_context_gib < 0 or
            not math.isfinite(reserve_gib) or reserve_gib < 0):
        raise ValueError("invalid resource budget")
    available = snapshot["ram_bytes"]["available"]
    required = (estimated_weights_bytes +
                independent_sessions * int(kv_context_gib * GIB) +
                int(reserve_gib * GIB))
    return {
        "estimated_model_weight_bytes": estimated_weights_bytes,
        "sessions": independent_sessions,
        "assumed_per_session_kv_gib": kv_context_gib,
        "assumed_system_reserve_gib": reserve_gib,
        "available_ram_bytes": available,
        "planned_ram_need_bytes": required,
        "ram_memory_check_passed": required <= available,
        "gpu_native_mxfp4_supported":
            any(gpu["hf_native_mxfp4_supported_by_cc"]
                for gpu in snapshot["gpus"]),
        "runtime_loaded": False,
        "tokens_per_second_measured": False,
        "warning": "Estimates only: activation, allocator, offload, VRAM, CPU speed and model quality not validated.",
    }
