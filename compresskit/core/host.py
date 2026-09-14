"""What this specific CPU can actually do.

INT8 only beats FP32 on latency when the CPU has fused low-precision dot-product
instructions (x86 VNNI/AMX, ARM DotProd/I8MM). Without them INT8 is routinely
*slower* than FP32 while still reporting a 4x smaller file, which is how
compression tools end up shipping regressions. So we probe the host up front and
let that gate the precision choice instead of inferring speed from size.
"""

from __future__ import annotations

import platform
import re
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache

# ISA features that give INT8 a fused dot-product/matmul path. Presence of any
# one of these is what separates "INT8 will be faster" from "benchmark it first".
FAST_INT8_FEATURES = ("amx_int8", "avx512_vnni", "avx_vnni", "i8mm", "dotprod")

# Canonical names we normalise every platform's spelling onto.
_X86_ALIASES = {
    "avx2": "avx2",
    "avx512f": "avx512f",
    "avx512_vnni": "avx512_vnni",
    "avx512vnni": "avx512_vnni",
    "avx_vnni": "avx_vnni",
    "avxvnni": "avx_vnni",
    "amx_int8": "amx_int8",
    "amxint8": "amx_int8",
    "f16c": "f16c",
}
_ARM_ALIASES = {
    "asimd": "neon",
    "neon": "neon",
    "asimddp": "dotprod",
    "dotprod": "dotprod",
    "feat_dotprod": "dotprod",
    "i8mm": "i8mm",
    "feat_i8mm": "i8mm",
    "bf16": "bf16",
    "feat_bf16": "bf16",
    "asimdhp": "fp16",
    "fphp": "fp16",
    "neon_fp16": "fp16",
}


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout if out.returncode == 0 else ""


def _normalise(tokens) -> set[str]:
    found = set()
    for raw in tokens:
        key = raw.strip().lower().replace(".", "_").replace("-", "_")
        if key in _X86_ALIASES:
            found.add(_X86_ALIASES[key])
        elif key in _ARM_ALIASES:
            found.add(_ARM_ALIASES[key])
    return found


def _numpy_features() -> set[str]:
    """Cross-platform CPUID baseline. Present on every OS numpy builds for."""
    try:
        from numpy._core._multiarray_umath import __cpu_features__ as feats
    except Exception:
        return set()
    return _normalise(name for name, enabled in feats.items() if enabled)


def _darwin_features() -> set[str]:
    found = set()
    # Apple silicon exposes one sysctl per architectural feature.
    listing = _run(["sysctl", "-a"])
    for match in re.finditer(r"hw\.optional\.(?:arm\.)?(\w+):\s*1", listing):
        found |= _normalise([match.group(1)])
    # Intel Macs report a flat feature string instead.
    for key in ("machdep.cpu.features", "machdep.cpu.leaf7_features"):
        found |= _normalise(_run(["sysctl", "-n", key]).split())
    return found


def _linux_features() -> set[str]:
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return set()
    found = set()
    for line in text.splitlines():
        if line.split(":")[0].strip() in ("flags", "Features"):
            found |= _normalise(line.split(":", 1)[1].split())
    return found


def _cpu_name() -> str:
    system = platform.system()
    if system == "Darwin":
        name = _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
        if name:
            return name
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine() or "unknown"


def _core_counts() -> tuple[int, int]:
    """(logical cores, performance cores). Falls back to logical when unsplit."""
    import os

    logical = os.cpu_count() or 1
    if platform.system() == "Darwin":
        # perflevel0 is the performance cluster on Apple silicon.
        raw = _run(["sysctl", "-n", "hw.perflevel0.logicalcpu"]).strip()
        if raw.isdigit():
            return logical, int(raw)
    return logical, logical


def _versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for label, module in (("torch", "torch"), ("onnxruntime", "onnxruntime"), ("numpy", "numpy")):
        try:
            versions[label] = __import__(module).__version__
        except Exception:
            versions[label] = "not installed"
    return versions


@dataclass(frozen=True)
class HostContext:
    """Attached to every measurement so numbers are never read without hardware."""

    system: str
    machine: str
    cpu: str
    logical_cores: int
    performance_cores: int
    features: frozenset[str]
    versions: dict[str, str] = field(default_factory=dict)

    @property
    def int8_features(self) -> list[str]:
        return [f for f in FAST_INT8_FEATURES if f in self.features]

    @property
    def int8_speed(self) -> str:
        """'fast' | 'slow' | 'unknown' — whether INT8 has a fused kernel path."""
        if self.int8_features:
            return "fast"
        if not self.features:
            return "unknown"
        return "slow"

    @property
    def recommended_precision(self) -> str:
        """Default precision for this host. INT8 only where it will pay off."""
        return "int8" if self.int8_speed == "fast" else "fp16"

    @property
    def reduce_range(self) -> bool:
        """Quantize weights to 7 bits to dodge saturation on pre-VNNI x86.

        Only x86 without VNNI has the INT8 accumulator overflow problem this
        works around; on ARM it just throws away a bit of precision for nothing.
        """
        return self.machine.lower() in ("x86_64", "amd64", "i386") and "avx512_vnni" not in self.features and "avx_vnni" not in self.features

    def precision_rationale(self) -> str:
        if self.int8_speed == "fast":
            return f"INT8 has fused kernels here ({', '.join(self.int8_features)}) — expect a real latency win."
        if self.int8_speed == "slow":
            return "No VNNI/AMX/DotProd/I8MM on this CPU — INT8 may be SLOWER than FP32. Benchmark before shipping it."
        return "Could not read CPU features — INT8 speed is unverified on this host. Benchmark before shipping it."

    def as_dict(self) -> dict:
        return {
            "system": self.system,
            "machine": self.machine,
            "cpu": self.cpu,
            "logical_cores": self.logical_cores,
            "performance_cores": self.performance_cores,
            "features": sorted(self.features),
            "int8_speed": self.int8_speed,
            "versions": dict(self.versions),
        }

    def describe(self) -> str:
        lines = [
            f"host      : {self.cpu} ({self.machine}, {self.system})",
            f"cores     : {self.logical_cores} logical / {self.performance_cores} performance",
            f"ISA       : {', '.join(sorted(self.features)) or 'none detected'}",
            f"INT8      : {self.int8_speed} — {self.precision_rationale()}",
            f"default   : {self.recommended_precision}",
            f"versions  : " + ", ".join(f"{k} {v}" for k, v in self.versions.items()),
        ]
        return "\n".join(lines)


def training_device() -> str:
    """Best available device for the fine-tune stages.

    Only training uses this. Every reported latency is CPU/ONNX Runtime, since
    that is the deployment target — an accelerator here just makes the
    distillation fine-tune finish sooner.
    """
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@lru_cache(maxsize=1)
def probe() -> HostContext:
    """Detect the host once per process; the answer cannot change under us."""
    system = platform.system()
    features = _numpy_features()
    if system == "Darwin":
        features |= _darwin_features()
    elif system == "Linux":
        features |= _linux_features()

    logical, performance = _core_counts()
    return HostContext(
        system=system,
        machine=platform.machine(),
        cpu=_cpu_name(),
        logical_cores=logical,
        performance_cores=performance,
        features=frozenset(features),
        versions=_versions(),
    )
