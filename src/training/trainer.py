"""Real CPU-first training selected solely by three-class validation Macro F1."""
from __future__ import annotations

import logging
import math
import platform
import random
import time
from collections.abc import Mapping

import numpy as np
import torch
from torch import nn


_LOG = logging.getLogger(__name__)
_DEFAULTS = {"seed": 42, "device": "cpu", "batch_size": 64, "max_epochs": 50,
             "learning_rate": .001, "weight_decay": .0001,
             "gradient_clip_norm": 1., "early_stopping_patience": 8,
             "class_weights": False}


def seed_training(seed: int) -> None:
    """Seed before model construction as well as fit for reproducible initialization."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _plain(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _metadata_rows(metadata, count: int) -> list[dict]:
    if metadata is None:
        return [{} for _ in range(count)]
    metadata = _plain(metadata)
    if isinstance(metadata, list) and len(metadata) == count and all(isinstance(row, dict) for row in metadata):
        return metadata
    if isinstance(metadata, Mapping):
        return [{key: value[i] if isinstance(value, list) and len(value) == count else value
                 for key, value in metadata.items()} for i in range(count)]
    raise ValueError("Batch metadata must be collated fields or one dictionary per window")


class ModelTrainer:
    """Optimize on train only; select and restore best validation CPU weights.

    Loaders must be re-iterable and include all accepted rows (drop_last=False).
    The preflight support pass prevents optimizer updates on blocked folds.
    config['val_coverage'] may supply scheduled/accepted/rejected from the shared
    index; otherwise coverage explicitly describes accepted loader rows only.
    No test loader is accepted or consulted.
    """

    def __init__(self, model: nn.Module, train_loader, val_loader, config: dict):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = {**_DEFAULTS, **dict(config)}
        for key in ("max_epochs", "early_stopping_patience", "batch_size"):
            value = self.config[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        seed = self.config["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 32:
            raise ValueError("seed must be an integer in [0,2**32)")
        for key in ("learning_rate", "gradient_clip_norm", "weight_decay"):
            value = self.config[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (key != "weight_decay" and value == 0):
                raise ValueError(f"{key} must be finite and {'nonnegative' if key == 'weight_decay' else 'positive'}")
        if not isinstance(self.config["class_weights"], bool):
            raise ValueError("class_weights must be a boolean; weights are derived from train counts only")
        self.device = torch.device(self.config["device"])
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Only CPU or explicitly available CUDA devices are supported")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable; CPU is the default")
        self.config["device"] = str(self.device)

    def _batch(self, batch):
        if not isinstance(batch, (list, tuple)) or len(batch) not in (2, 3):
            raise ValueError("Loader must yield (windows, labels[, metadata])")
        x = torch.as_tensor(batch[0], dtype=torch.float32, device=self.device)
        raw_labels = torch.as_tensor(batch[1], device=self.device)
        if x.ndim != 3 or x.shape[1] != 100 or len(x) == 0:
            raise ValueError("Loader windows must have nonempty shape [B,100,F]")
        if not torch.isfinite(x).all():
            raise ValueError("Loader windows must contain finite scaled values")
        if raw_labels.ndim != 1 or len(raw_labels) != len(x) or not torch.isfinite(raw_labels).all() or not torch.isin(raw_labels, raw_labels.new_tensor([0, 1, 2])).all():
            raise ValueError("Loader labels must be class IDs 0/1/2 matching windows")
        labels = raw_labels.long()
        return x, labels, batch[2] if len(batch) == 3 else None

    def _support(self, loader, role: str) -> np.ndarray:
        if getattr(loader, "drop_last", False):
            raise ValueError(f"{role} loader must not drop accepted windows")
        if iter(loader) is loader:
            raise ValueError(f"{role} loader must be re-iterable")
        counts = np.zeros(3, dtype=np.int64)
        for batch in loader:
            _, labels, _ = self._batch(batch)
            counts += torch.bincount(labels, minlength=3).cpu().numpy()
        if counts.sum() == 0:
            raise ValueError(f"{role} loader is empty")
        if (counts == 0).any():
            raise ValueError(f"All three {role} classes 0/1/2 are required; counts={counts.tolist()}")
        return counts

    def fit(self) -> dict:
        # Evaluator is imported only when training is requested, not at import
        # time. The metrics core owns the definition and undefined-support rules.
        from src.evaluation.evaluator import ModelEvaluator

        started = time.perf_counter()
        seed_training(self.config["seed"])
        for loader in (self.train_loader, self.val_loader):
            generator = getattr(loader, "generator", None)
            if generator is not None:
                generator.manual_seed(self.config["seed"])
        _LOG.info("Training seed=%s device=%s deterministic_algorithms=True", self.config["seed"], self.device)
        if isinstance(getattr(self.val_loader, "sampler", None), torch.utils.data.RandomSampler):
            raise ValueError("validation loader must preserve deterministic chronological order, without shuffle")
        train_counts = self._support(self.train_loader, "training")
        val_counts = self._support(self.val_loader, "validation")
        val_n = int(val_counts.sum())
        coverage = dict(self.config.get("val_coverage", {"scheduled": val_n, "accepted": val_n, "rejected": 0}))
        evaluation_metadata = {"class_order": [0, 1, 2], "split": "validation",
                               "coverage_scope": "scheduled_index" if "val_coverage" in self.config else "accepted_loader_rows"}
        # Validate external coverage before any optimizer update.
        if any(isinstance(coverage.get(key), bool) or not isinstance(coverage.get(key), int) or coverage[key] < 0
               for key in ("scheduled", "accepted", "rejected")) or coverage["accepted"] != val_n or coverage["scheduled"] != val_n + coverage["rejected"]:
            raise ValueError("validation coverage must match accepted rows and scheduled=accepted+rejected")
        weights = None
        if self.config["class_weights"]:
            weights = torch.tensor(train_counts.sum() / (3. * train_counts), dtype=torch.float32, device=self.device)
        self.model.to(self.device)
        criterion = nn.CrossEntropyLoss(weight=weights, reduction="sum")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"],
                                     weight_decay=self.config["weight_decay"])
        evaluator = ModelEvaluator()
        history = []
        best_score = -math.inf
        best_epoch = 0
        best_state = None
        best_metrics = None
        best_predictions = None
        stale = 0
        for epoch in range(1, self.config["max_epochs"] + 1):
            epoch_started = time.perf_counter()
            self.model.train()
            train_total = train_denominator = 0.
            train_seen = 0
            for batch in self.train_loader:
                x, labels, _ = self._batch(batch)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(x)
                if logits.shape != (len(labels), 3) or not torch.isfinite(logits).all():
                    raise ValueError("Training logits must be finite [B,3]")
                loss_sum = criterion(logits, labels)
                denominator = len(labels) if weights is None else float(weights[labels].sum().item())
                loss = loss_sum / denominator
                if not torch.isfinite(loss):
                    raise ValueError("Training loss must be finite")
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config["gradient_clip_norm"], error_if_nonfinite=True)
                optimizer.step()
                train_total += float(loss_sum.detach().item())
                train_denominator += denominator
                train_seen += len(labels)
            if train_seen != int(train_counts.sum()):
                raise ValueError("training loader accepted support changed between passes")
            self.model.eval()
            val_total = val_denominator = 0.
            labels_parts, probability_parts, windows = [], [], []
            with torch.inference_mode():
                for batch in self.val_loader:
                    x, labels, metadata = self._batch(batch)
                    logits = self.model(x)
                    if logits.shape != (len(labels), 3) or not torch.isfinite(logits).all():
                        raise ValueError("Validation logits must be finite [B,3]")
                    loss_sum = criterion(logits, labels)
                    probabilities = logits.softmax(dim=1)
                    if not torch.isfinite(loss_sum) or not torch.isfinite(probabilities).all():
                        raise ValueError("Validation loss/probabilities must be finite")
                    val_total += float(loss_sum.item())
                    val_denominator += len(labels) if weights is None else float(weights[labels].sum().item())
                    labels_parts.append(labels.cpu().numpy())
                    probability_parts.append(probabilities.cpu().numpy())
                    windows.extend(_metadata_rows(metadata, len(labels)))
            if not labels_parts:
                raise ValueError("validation loader became empty")
            labels_np = np.concatenate(labels_parts)
            probabilities_np = np.concatenate(probability_parts)
            if not np.array_equal(np.bincount(labels_np, minlength=3), val_counts):
                raise ValueError("validation accepted class support changed between passes")
            metrics = evaluator.evaluate(labels_np, probabilities_np,
                                         metadata={**evaluation_metadata, "windows": windows}, coverage=coverage)
            score = metrics.get("macro_f1")
            if score is None or not math.isfinite(score):
                raise ValueError("validation three-class Macro F1 is undefined or nonfinite")
            train_loss = train_total / train_denominator
            val_loss = val_total / val_denominator
            if not math.isfinite(train_loss) or not math.isfinite(val_loss):
                raise ValueError("Epoch losses must be finite")
            improved = score > best_score
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                            "val_macro_f1": float(score), "val_accuracy": metrics["accuracy"],
                            "val_weighted_f1": metrics["weighted_f1"],
                            "val_balanced_accuracy": metrics["balanced_accuracy"],
                            "train_samples": train_seen, "val_samples": len(labels_np),
                            "elapsed_s": time.perf_counter() - epoch_started, "selected": improved})
            _LOG.info("Epoch=%d train_loss=%.6f val_loss=%.6f val_macro_f1=%.6f selected=%s",
                      epoch, train_loss, val_loss, score, improved)
            if improved:
                best_score = float(score)
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                best_metrics = metrics
                best_predictions = {"labels": labels_np.copy(), "probabilities": probabilities_np.copy(), "metadata": windows}
                stale = 0
            else:
                stale += 1
            if stale >= self.config["early_stopping_patience"]:
                break
        self.model.load_state_dict(best_state)
        self.model.eval()
        return {"history": history, "best_epoch": best_epoch, "best_val_metrics": best_metrics,
                "state_dict": best_state, "best_val_predictions": best_predictions,
                "selection_metric": "val_macro_f1", "selection_tie_policy": "first_epoch",
                "config": dict(self.config), "class_weights": None if weights is None else weights.cpu().tolist(),
                "train_class_counts": train_counts.tolist(), "val_class_counts": val_counts.tolist(),
                "stopped_early": len(history) < self.config["max_epochs"],
                "elapsed_s": time.perf_counter() - started,
                "environment": {"python": platform.python_version(), "platform": platform.platform(),
                                "torch": str(torch.__version__), "numpy": np.__version__,
                                "device": str(self.device), "cuda_available": torch.cuda.is_available(),
                                "train_batch_size": getattr(self.train_loader, "batch_size", None),
                                "val_batch_size": getattr(self.val_loader, "batch_size", None),
                                "train_num_workers": getattr(self.train_loader, "num_workers", None),
                                "val_num_workers": getattr(self.val_loader, "num_workers", None),
                                "seed": self.config["seed"], "deterministic_algorithms": True,
                                "seed_scope": "fit; call seed_training before constructing model for initialization"}}
