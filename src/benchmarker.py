"""Batch-size-one workstation benchmark for a calibrated EfficientAD checkpoint."""

from __future__ import annotations

import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from trainer import build_from_checkpoint, category_output


def benchmark(
    config: dict[str, Any], category: str, seed: int, device: torch.device, normal_loader: DataLoader
) -> Path:
    wrapper, payload, _ = build_from_checkpoint(config, category, seed, device)
    if not payload.get("calibrated"):
        raise RuntimeError("Benchmark requires a calibrated checkpoint")
    wrapper.core.eval()
    batch = next(iter(normal_loader))
    image = batch.image.to(device)
    if image.shape[0] != 1:
        raise RuntimeError("Benchmark input must use batch size 1")

    settings = config["benchmark"]
    warmup = int(settings["warmup_iterations"])
    iterations = int(settings["measured_iterations"])
    with torch.inference_mode():
        for _ in range(warmup):
            wrapper.core(image)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    latencies: list[float] = []
    with torch.inference_mode():
        for _ in range(iterations):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                wrapper.core(image)
                end.record()
                end.synchronize()
                latencies.append(float(start.elapsed_time(end)))
            else:
                started = time.perf_counter()
                wrapper.core(image)
                latencies.append((time.perf_counter() - started) * 1000.0)

    mean_latency = statistics.fmean(latencies)
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "category": category,
        "seed": seed,
        "device": str(device),
        "batch_size": 1,
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "latency_ms_mean": mean_latency,
        "latency_ms_p50": float(np.percentile(latencies, 50)),
        "latency_ms_p95": float(np.percentile(latencies, 95)),
        "throughput_images_per_second": 1000.0 / mean_latency,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
        ),
        "reference_only": {"paper_latency_ms": 2.0, "paper_throughput_images_per_second": 600.0},
    }
    output = category_output(config, category) / f"benchmark_seed_{seed}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return output
