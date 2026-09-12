"""The training loop.

Sized for the 4 GB budget in PLAN.md §3.2: mixed precision, a small per-step
batch with gradient accumulation to reach a useful effective batch, and
channels-last memory format. Nothing here is model-specific -- the loop reads
whatever keys the model returns and hands them to the loss, so an instance head
or a query decoder slots in without touching this file.
"""
from __future__ import annotations

import time
from dataclasses import asdict

import numpy as np
import torch

from ..config import N_CLASSES
from ..eval.evaluate import evaluate_model
from ..eval.metrics_sem import SemanticMetrics
from .config import TrainConfig
from .tracking import Run


def cosine_with_warmup(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))


class Trainer:
    def __init__(self, cfg: TrainConfig, model, loss_fn, train_loader, val_loader,
                 device, run: Run):
        self.cfg, self.model, self.loss_fn = cfg, model, loss_fn
        self.train_loader, self.val_loader = train_loader, val_loader
        self.device, self.run = device, run

        self.model.to(device, memory_format=torch.channels_last)
        self.optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr,
                                           weight_decay=cfg.optim.weight_decay)
        self.use_amp = cfg.optim.amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        steps_per_epoch = max(1, len(train_loader) // cfg.data.accum_steps)
        self.total_steps = steps_per_epoch * cfg.optim.epochs
        self.warmup_steps = int(cfg.optim.warmup_frac * self.total_steps)
        self.global_step = 0
        self.best_miou = -1.0

    # -- one epoch ----------------------------------------------------------
    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        if hasattr(self.train_loader.dataset, "set_epoch"):
            self.train_loader.dataset.set_epoch(epoch)

        accum = self.cfg.data.accum_steps
        running, n_batches, started = {}, 0, time.time()
        self.optimiser.zero_grad(set_to_none=True)

        for i, batch in enumerate(self.train_loader):
            views = batch["views"].to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model({"views": views})
                loss, parts = self.loss_fn(outputs, batch)
            self.scaler.scale(loss / accum).backward()

            for key, value in parts.items():
                running[key] = running.get(key, 0.0) + value
            n_batches += 1

            if (i + 1) % accum == 0:
                scale = cosine_with_warmup(self.global_step, self.total_steps,
                                           self.warmup_steps)
                for group in self.optimiser.param_groups:
                    group["lr"] = self.cfg.optim.lr * scale

                if self.cfg.optim.grad_clip > 0:
                    self.scaler.unscale_(self.optimiser)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.cfg.optim.grad_clip)
                self.scaler.step(self.optimiser)
                self.scaler.update()
                self.optimiser.zero_grad(set_to_none=True)
                self.global_step += 1

                if self.global_step % self.cfg.run.log_every == 0:
                    self.run.scalar("train/loss", running["loss"] / n_batches, self.global_step)
                    self.run.scalar("train/lr", self.cfg.optim.lr * scale, self.global_step)

        return {k: v / max(n_batches, 1) for k, v in running.items()} | {
            "epoch_seconds": time.time() - started}

    # -- full run -----------------------------------------------------------
    def fit(self) -> dict:
        history = []
        for epoch in range(self.cfg.optim.epochs):
            train_stats = self.train_epoch(epoch)
            metrics, _ = evaluate_model(self.model, self.val_loader, self.device,
                                        amp=self.use_amp)
            summary = metrics.summary("exclusive")
            foreground = metrics.foreground_summary("exclusive")

            row = {
                "epoch": epoch,
                "train_loss": round(train_stats.get("loss", float("nan")), 5),
                "train_focal": round(train_stats.get("focal", float("nan")), 5),
                "train_dice": round(train_stats.get("dice", float("nan")), 5),
                "val_miou": round(summary["miou"], 5),
                "val_pixel_acc": round(summary["pixel_accuracy"], 5),
                "val_fg_iou": round(foreground["iou"], 5),
                "seconds": round(train_stats["epoch_seconds"], 1),
                "lr": self.optimiser.param_groups[0]["lr"],
            }
            history.append(row)
            self.run.log_row(row)
            for key in ("val_miou", "val_pixel_acc", "val_fg_iou"):
                self.run.scalar(key.replace("_", "/", 1), row[key], self.global_step)

            print(f"  epoch {epoch:3d}  loss {row['train_loss']:.4f}  "
                  f"val mIoU {row['val_miou']:.4f}  fg IoU {row['val_fg_iou']:.4f}  "
                  f"{row['seconds']:.0f}s", flush=True)

            self.save_checkpoint("last.pt", epoch, row)
            if summary["miou"] > self.best_miou:
                self.best_miou = summary["miou"]
                self.save_checkpoint("best.pt", epoch, row)
                self.run.save_json("best_metrics.json", metrics.to_dict())

        self.run.save_json("history.json", history)
        return {"best_miou": self.best_miou, "history": history}

    def save_checkpoint(self, name: str, epoch: int, row: dict) -> None:
        torch.save({
            "model": self.model.state_dict(),
            "optimiser": self.optimiser.state_dict(),
            "epoch": epoch,
            "global_step": self.global_step,
            "metrics": row,
            "config": asdict(self.cfg),
            "torch_rng": torch.get_rng_state(),
        }, self.run.dir / name)


def load_checkpoint(path, device=None):
    """Rebuild a model from a checkpoint written by `Trainer.save_checkpoint`."""
    from ..models.registry import build_model
    from .config import TrainConfig
    from ..config import _build

    payload = torch.load(path, map_location=device or "cpu", weights_only=False)
    cfg = _build(TrainConfig, payload["config"])
    model = build_model(cfg.model.name, n_classes=N_CLASSES,
                        base_width=cfg.model.base_width, depth=cfg.model.depth,
                        in_channels=cfg.model.in_channels)
    model.load_state_dict(payload["model"])
    if device is not None:
        model.to(device)
    return model, cfg, payload
