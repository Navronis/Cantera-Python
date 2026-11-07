"""Numerically honest native-versus-Python comparison metrics.

The routines in this module are shared by oracle, convergence, and regression
tools.  They keep absolute, relative, mixed-tolerance, ULP, and logarithmic
errors together so a near-zero quantity cannot be misreported from relative
error alone.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import struct
from typing import Iterable


@dataclass(frozen=True)
class ErrorMetrics:
    python_value: float
    native_value: float
    absolute_error: float
    relative_error: float
    mixed_error: float
    ulp_distance: int | None
    log10_error: float | None
    meaningful_relative_error: float | None
    near_zero: bool
    passed: bool

    def as_dict(self) -> dict:
        return asdict(self)


def _ordered_float64(value: float) -> int:
    """Map finite IEEE-754 binary64 values to monotonically ordered integers."""
    bits = struct.unpack(">Q", struct.pack(">d", float(value)))[0]
    if bits & (1 << 63):
        return (~bits) & ((1 << 64) - 1)
    return bits | (1 << 63)


def ulp_distance(left: float, right: float) -> int | None:
    """Return binary64 representable-value distance, or ``None`` for NaN/inf."""
    left = float(left)
    right = float(right)
    if left == right:
        return 0
    if not (math.isfinite(left) and math.isfinite(right)):
        return None
    return abs(_ordered_float64(left) - _ordered_float64(right))


def compare_scalar(python_value: float, native_value: float, *,
                   atol: float, rtol: float, relative_floor: float = 1e-300,
                   near_zero_floor: float | None = None) -> ErrorMetrics:
    """Compare two scalar float64 values using all required error measures."""
    if atol < 0.0 or rtol < 0.0 or relative_floor <= 0.0:
        raise ValueError("atol/rtol must be non-negative and relative_floor positive")
    python_value = float(python_value)
    native_value = float(native_value)
    near_zero_floor = (max(float(atol), float(relative_floor))
                       if near_zero_floor is None else float(near_zero_floor))
    if near_zero_floor < 0.0:
        raise ValueError("near_zero_floor must be non-negative")

    if python_value == native_value:
        absolute = relative = mixed = 0.0
    elif not (math.isfinite(python_value) and math.isfinite(native_value)):
        absolute = relative = mixed = math.inf
    else:
        absolute = abs(python_value - native_value)
        relative = absolute / max(abs(native_value), relative_floor)
        denominator = atol + rtol * abs(native_value)
        mixed = absolute / denominator if denominator > 0.0 else (
            0.0 if absolute == 0.0 else math.inf)

    near_zero = abs(native_value) <= near_zero_floor
    meaningful_relative = None if near_zero else relative
    log_error = None
    if (python_value > 0.0 and native_value > 0.0
            and math.isfinite(python_value) and math.isfinite(native_value)):
        log_error = abs(math.log10(python_value) - math.log10(native_value))
    passed = mixed <= 1.0
    return ErrorMetrics(
        python_value=python_value,
        native_value=native_value,
        absolute_error=absolute,
        relative_error=relative,
        mixed_error=mixed,
        ulp_distance=ulp_distance(python_value, native_value),
        log10_error=log_error,
        meaningful_relative_error=meaningful_relative,
        near_zero=near_zero,
        passed=passed,
    )


def compare_sequence(python_values: Iterable[float],
                     native_values: Iterable[float], **kwargs) -> list[ErrorMetrics]:
    python_values = list(python_values)
    native_values = list(native_values)
    if len(python_values) != len(native_values):
        raise ValueError("comparison sequences must have the same length")
    return [compare_scalar(python, native, **kwargs)
            for python, native in zip(python_values, native_values)]


def summarize(metrics: Iterable[ErrorMetrics]) -> dict:
    metrics = list(metrics)
    meaningful = [item.meaningful_relative_error for item in metrics
                  if item.meaningful_relative_error is not None]
    ulps = [item.ulp_distance for item in metrics if item.ulp_distance is not None]
    logs = [item.log10_error for item in metrics if item.log10_error is not None]
    return {
        "count": len(metrics),
        "passed": sum(item.passed for item in metrics),
        "failed": sum(not item.passed for item in metrics),
        "near_zero_count": sum(item.near_zero for item in metrics),
        "max_absolute_error": max((item.absolute_error for item in metrics), default=0.0),
        "max_relative_error": max((item.relative_error for item in metrics), default=0.0),
        "max_meaningful_relative_error": max(meaningful, default=None),
        "max_mixed_error": max((item.mixed_error for item in metrics), default=0.0),
        "max_ulp_distance": max(ulps, default=None),
        "max_log10_error": max(logs, default=None),
    }
