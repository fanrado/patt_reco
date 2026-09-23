"""The training loop.

Deliberately plain: AdamW, cross-entropy, a cosine schedule and gradient
clipping. No mixed precision, no gradient accumulation, no distributed
support -- the dataset is small and the model is tiny, so none of it would
buy anything.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .metrics import accuracy, roc_auc
from .tracking import Run


class Trainer:
    """Trains one model against one dataset, reporting through a `Run`."""

    def __init__(self, cfg, model, train_loader, val_loader, device, run: Run):
        self.cfg = cfg
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.run = run

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.optim.lr,
            weight_decay=cfg.optim.weight_decay)

        total_steps = max(1, cfg.optim.epochs * max(1, len(train_loader)))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps)

    # -- one epoch ---------------------------------------------------------
    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total, seen = 0.0, 0

        for step, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            loss = self.criterion(self.model(images), labels)
            loss.backward()
            if self.cfg.optim.grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.cfg.optim.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

            total += loss.item() * len(labels)
            seen += len(labels)
            if self.cfg.run.log_every and step % self.cfg.run.log_every == 0:
                print(f"  epoch {epoch:3d}  step {step:5d}/{len(self.train_loader)}  "
                      f"loss {loss.item():.4f}  lr {self.scheduler.get_last_lr()[0]:.2e}")

        return total / max(seen, 1)

    # -- evaluation --------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, loader):
        """Return (loss, y_true, y_pred, scores); scores are P(class 1)."""
        self.model.eval()
        total, seen = 0.0, 0
        y_true, y_pred, scores = [], [], []

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)
            total += self.criterion(logits, labels).item() * len(labels)
            seen += len(labels)

            y_true.append(labels.cpu().numpy())
            y_pred.append(logits.argmax(dim=1).cpu().numpy())
            scores.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())

        if not y_true:
            empty = np.empty(0)
            return float("nan"), empty, empty, empty
        return (total / max(seen, 1), np.concatenate(y_true),
                np.concatenate(y_pred), np.concatenate(scores))

    # -- the run -----------------------------------------------------------
    def train(self) -> float:
        """Run every epoch, checkpointing on the best validation accuracy."""
        best_acc = -1.0

        for epoch in range(self.cfg.optim.epochs):
            train_loss = self._train_epoch(epoch)
            val_loss, y_true, y_pred, scores = self.evaluate(self.val_loader)
            val_acc = accuracy(y_true, y_pred)
            val_auc = roc_auc(y_true, scores)

            self.run.log(epoch, train_loss=train_loss, val_loss=val_loss,
                         val_acc=val_acc, val_auc=val_auc)
            self.run.save_last(self.model)

            improved = val_acc > best_acc
            if improved:
                best_acc = val_acc
                self.run.save_best(self.model)

            print(f"epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}  "
                  f"acc {val_acc:.4f}  auc {val_auc:.4f}"
                  f"{'  <- best' if improved else ''}")

        return best_acc

    # -- wiring check ------------------------------------------------------
    def overfit_one_batch(self, steps: int = 200) -> float:
        """Drive a single batch to near-zero loss.

        With one convolution this cannot fail for lack of capacity, so a loss
        that refuses to fall means the plumbing is wrong -- labels detached
        from images, a frozen parameter, a dead learning rate.
        """
        images, labels = next(iter(self.train_loader))
        images = images.to(self.device)
        labels = labels.to(self.device)

        self.model.train()
        loss = torch.tensor(float("nan"))
        for step in range(steps):
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(images), labels)
            loss.backward()
            self.optimizer.step()
            if step % 20 == 0:
                print(f"  overfit step {step:4d}  loss {loss.item():.6f}")

        final = loss.item()
        print(f"  overfit final    loss {final:.6f}")
        return final
