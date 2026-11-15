"""JSONL evidence and deterministic rules; no model has execution privileges."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch
from torch import nn


def gradient_census(modules: dict[str, nn.Module]) -> dict[str, Any]:
    parameters: dict[str, Any] = {}
    summaries: dict[str, Any] = {}
    for module_name, module in modules.items():
        active = count = elements = nonzero = 0
        square_norm = 0.0
        for name, parameter in module.named_parameters():
            grad = parameter.grad
            norm = float(grad.norm().item()) if grad is not None else 0.0
            nz = int(torch.count_nonzero(grad).item()) if grad is not None else 0
            parameters[f"{module_name}.{name}"] = {
                "requires_grad": parameter.requires_grad, "missing": grad is None,
                "finite": grad is None or bool(torch.isfinite(grad).all().item()),
                "norm": norm, "nonzero_fraction": nz / parameter.numel(),
                "elements": parameter.numel(),
            }
            if parameter.requires_grad:
                count += 1
                active += int(nz > 0)
                elements += parameter.numel()
                nonzero += nz
                square_norm += norm * norm
        summaries[module_name] = {
            "trainable_tensors": count, "active_tensor_fraction": active / count if count else None,
            "nonzero_element_fraction": nonzero / elements if elements else None,
            "norm": math.sqrt(square_norm),
        }
    return {"parameters": parameters, "modules": summaries}


class RuleMonitor:
    """Rules produce evidence; the runner alone honors their explicit stop action."""

    def __init__(self, configuration: dict[str, Any] | None = None) -> None:
        self.configuration = dict(configuration) if configuration is not None else None
        if self.configuration is not None:
            required = {"version", "data_ratio", "patience", "minimum_floor",
                        "ramp_end_epoch", "required_post_ramp_epochs"}
            if (set(self.configuration) != required or self.configuration["version"] != 2
                    or self.configuration["data_ratio"] <= 1
                    or self.configuration["patience"] < 1
                    or self.configuration["minimum_floor"] <= 0
                    or self.configuration["ramp_end_epoch"] < 0
                    or self.configuration["required_post_ramp_epochs"] < 1
                    or not all(math.isfinite(float(v)) for v in self.configuration.values())):
                raise ValueError("invalid preregistered monitor configuration")
        self.initial: dict[str, float] = {}
        self.dead_counts: dict[tuple[str, str], int] = {}
        self.timings: dict[str, list[float]] = {}
        self.data_minima: dict[tuple[str, str, str], tuple[float, int]] = {}
        self.data_counts: dict[tuple[str, str, str], int] = {}
        self.last_epochs: dict[str, int] = {}
        self.weight_baselines: dict[tuple[str, float], float] = {}

    def check(self, record: dict[str, Any], line: int) -> list[dict[str, Any]]:
        alerts: list[dict[str, Any]] = []
        stage, epoch = str(record["stage"]), int(record["epoch"])

        def alert(rule: str, field: str, value: Any, threshold: Any, stop: bool) -> None:
            alerts.append({
                "rule": rule, "level": "blocking" if stop else "warning",
                "action": "stop" if stop else "observe", "stage": stage, "epoch": epoch,
                "evidence": {"metrics_line": line, "field": field,
                             "value": value, "threshold": threshold},
            })

        def check_finite(value: Any, field: str) -> None:
            if isinstance(value, float) and not math.isfinite(value):
                alert("nonfinite", field, str(value), "finite", True)
            elif isinstance(value, dict):
                for key, child in value.items():
                    check_finite(child, f"{field}.{key}")
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    check_finite(child, f"{field}.{index}")

        check_finite(record, "metrics")
        total = float(record["total"])
        self.initial.setdefault(stage, max(total, 1e-12))
        if self.configuration is None and epoch > 30 and total > 10 * self.initial[stage]:
            alert("divergence", "total", total, 10 * self.initial[stage], True)
        if self.configuration is not None:
            config = self.configuration
            weight = float(record.get("physics_weight", 0.0))
            if math.isfinite(total) and math.isfinite(weight):
                baseline = self.weight_baselines.setdefault((stage, weight), total)
                record["same_weight_total_comparison"] = {
                    "physics_weight": weight, "baseline": baseline,
                    "ratio": total / max(baseline, config["minimum_floor"]),
                    "action": "diagnostic_only",
                }
            contiguous = self.last_epochs.get(stage, epoch - 1) == epoch - 1
            self.last_epochs[stage] = epoch
            if set(record.get("protocols", {})) != {"K", "U"}:
                alert("metric_contract", "protocols", list(record.get("protocols", {})),
                      "both K and U", True)
            for protocol, metrics in record.get("protocols", {}).items():
                for name in ("imp", "obs", "coef"):
                    key = (stage, protocol, name)
                    support = metrics.get("support_counts", {}).get(name)
                    if (type(support) is not int or support < 0
                            or name not in metrics.get("normalized", {})):
                        alert("metric_contract", f"protocols.{protocol}.{name}", support,
                              "nonnegative integer support and normalized value", True)
                        continue
                    if support == 0:
                        self.data_counts[key] = 0
                        continue
                    current = float(metrics["normalized"][name])
                    if not math.isfinite(current):
                        continue  # The recursive finite rule already records the stop.
                    minimum, minimum_line = self.data_minima.get(key, (current, line))
                    if current < minimum:
                        minimum, minimum_line = current, line
                    self.data_minima[key] = (minimum, minimum_line)
                    threshold = config["data_ratio"] * max(minimum, config["minimum_floor"])
                    previous_count = self.data_counts.get(key, 0) if contiguous else 0
                    count = previous_count + 1 if current > threshold else 0
                    self.data_counts[key] = count
                    if count == config["patience"]:
                        alert("data_divergence", f"protocols.{protocol}.normalized.{name}",
                              current, threshold, True)
                        alerts[-1]["evidence"].update(
                            minimum=minimum, minimum_metrics_line=minimum_line,
                            minimum_floor=config["minimum_floor"], consecutive_epochs=count,
                        )
        drift = float(record.get("teacher_drift", 0.0))
        if drift != 0.0:
            alert("teacher_drift", "teacher_drift", drift, 0.0, True)
        if epoch > 2 and "gradients" in record:
            for name, values in record["gradients"]["parameters"].items():
                if not values["requires_grad"]:
                    continue
                key = (stage, name)
                self.dead_counts[key] = (
                    self.dead_counts.get(key, 0) + 1 if values["nonzero_fraction"] == 0 else 0
                )
                if self.dead_counts[key] == 5:
                    alert("dead_path", f"gradients.parameters.{name}", values, "5 epochs", True)
        seconds = float(record.get("epoch_seconds", 0.0))
        if epoch > 0 and seconds > 0:
            history = self.timings.setdefault(stage, [])
            if len(history) >= 5 and seconds > 3 * statistics.median(history[:5]):
                alert("epoch_time", "epoch_seconds", seconds,
                      3 * statistics.median(history[:5]), False)
            history.append(seconds)
        memory = float(record.get("peak_memory_bytes", 0))
        capacity = float(record.get("device_memory_bytes", 0))
        if capacity and memory > capacity * 0.9:
            alert("memory", "peak_memory_bytes", memory, capacity * 0.9, False)
        ratio = float(record.get("slack_explanation_ratio", 0.0))
        if ratio > 0.5:
            alert("slack_explanation", "slack_explanation_ratio", ratio, 0.5, False)
        return alerts


class EvidenceWriter:
    """Append metrics and alert evidence; consumer failures cannot stop training.

    A metrics disk failure is a reproducibility failure and is deliberately not
    swallowed. An alert observer is optional; a separate reader may tail JSONL.
    """

    def __init__(self, directory: Path, monitor_configuration: dict[str, Any] | None = None) -> None:
        self.directory = directory
        self.line = 0
        self.metrics = (directory / "metrics.jsonl").open("x", encoding="utf-8")
        self.alerts = (directory / "alerts.jsonl").open("x", encoding="utf-8")
        self.monitor = RuleMonitor(monitor_configuration)

    def write(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        self.line += 1
        record["metrics_line"] = self.line
        alerts = self.monitor.check(record, self.line)
        # On a nonfinite failure preserve exact evidence as strings: valid JSON.
        def sanitize(value: Any) -> Any:
            if isinstance(value, float) and not math.isfinite(value):
                return str(value)
            if isinstance(value, dict):
                return {k: sanitize(v) for k, v in value.items()}
            if isinstance(value, list):
                return [sanitize(v) for v in value]
            return value
        self.metrics.write(json.dumps(sanitize(record), ensure_ascii=False, allow_nan=False) + "\n")
        self.metrics.flush()
        for alert in alerts:
            self.write_alert(alert)
        return alerts

    def write_alert(self, alert: dict[str, Any]) -> None:
        self.alerts.write(json.dumps(alert, ensure_ascii=False, allow_nan=False) + "\n")
        self.alerts.flush()

    def close(self) -> None:
        self.metrics.close()
        self.alerts.close()
