"""Shared evaluation: run a model (or a classical baseline) over a loader."""
from __future__ import annotations

import numpy as np

from ..config import N_CLASSES
from .metrics_sem import SemanticMetrics


def evaluate_model(model, loader, device, amp: bool = True, max_batches: int = 0,
                   metrics: SemanticMetrics | None = None,
                   per_event: bool = False) -> tuple[SemanticMetrics, list]:
    """Accumulate semantic metrics over a loader. Returns (metrics, per-event rows)."""
    import torch

    metrics = metrics or SemanticMetrics(N_CLASSES)
    rows: list[dict] = []
    model.eval()

    autocast = torch.autocast(device_type=device.type,
                              enabled=amp and device.type == "cuda")
    with torch.no_grad():
        for step, batch in enumerate(loader):
            if max_batches and step >= max_batches:
                break
            views = batch["views"].to(device, non_blocking=True)
            with autocast:
                outputs = model({"views": views})
            pred = outputs["sem_logits"].argmax(dim=2).cpu().numpy()
            true = batch["semantic"].numpy()
            contrib = batch["n_contrib"].numpy()
            metrics.update(pred, true, contrib)

            if per_event:
                for b in range(pred.shape[0]):
                    single = SemanticMetrics(N_CLASSES)
                    single.update(pred[b], true[b], contrib[b])
                    rows.append({"index": int(batch["index"][b]),
                                 "miou": single.summary("exclusive")["miou"],
                                 "foreground_iou": single.foreground_summary("exclusive")["iou"]})
    return metrics, rows


def evaluate_baseline(baseline, dataset, max_events: int = 0,
                      metrics: SemanticMetrics | None = None,
                      per_event: bool = False) -> tuple[SemanticMetrics, list]:
    """Same, for a classical baseline that consumes raw ADC arrays."""
    metrics = metrics or SemanticMetrics(N_CLASSES)
    rows: list[dict] = []
    total = len(dataset) if not max_events else min(max_events, len(dataset))

    for i in range(total):
        record = dataset.reader[i]
        adc = record.adc(noise_cfg=dataset.noise_cfg, noise_scale=dataset.noise_scale)
        prediction = baseline.predict(adc)
        true = record.dense_semantic()
        contrib = record.dense_n_contrib()
        metrics.update(prediction.semantic, true, contrib)
        if per_event:
            single = SemanticMetrics(N_CLASSES)
            single.update(prediction.semantic, true, contrib)
            rows.append({"index": i,
                         "miou": single.summary("exclusive")["miou"],
                         "foreground_iou": single.foreground_summary("exclusive")["iou"]})
    return metrics, rows
