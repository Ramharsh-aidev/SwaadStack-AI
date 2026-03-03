"""Trainer class — training loop with early stopping, scheduling, gradient clipping."""

import time
from typing import Any, Dict, List

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from swaadstack.config import TrainingConfig, model_config
from swaadstack.models.swaadstack import SwaadStackModel
from swaadstack.training.metrics import RecommenderMetrics
from swaadstack.utils.helpers import print_banner, print_metrics, console


class Trainer:
    """Training orchestrator with early stopping, LR scheduling, and comprehensive metrics."""

    def __init__(self, model: SwaadStackModel, train_loader: DataLoader,
                 val_loader: DataLoader, config: TrainingConfig):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)

        self.optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

        total_steps = len(train_loader) * config.num_epochs
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=config.learning_rate,
            total_steps=total_steps, pct_start=0.1, anneal_strategy="cos",
        )

        self.metrics = RecommenderMetrics()
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.training_history: List[Dict[str, float]] = []

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        all_logits, all_labels = [], []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}", leave=True)
        for batch in pbar:
            cart_emb = batch["cart_embeddings"].to(self.device)
            target_emb = batch["target_embedding"].to(self.device)
            mask = batch["padding_mask"].to(self.device)
            temporal = batch["temporal_features"].to(self.device)
            geohash = batch["geohash_bucket"].to(self.device)
            labels = batch["label"].to(self.device)

            output = self.model(cart_embeddings=cart_emb, target_embeddings=target_emb,
                                padding_mask=mask, temporal_features=temporal,
                                geohash_buckets=geohash, labels=labels)
            loss = output["loss"]

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            all_logits.extend(output["logits"].detach().cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"})

        avg_loss = total_loss / max(num_batches, 1)
        auc = self.metrics.compute_auc(np.array(all_logits), np.array(all_labels))
        return {"train_loss": avg_loss, "train_auc": auc, "learning_rate": self.scheduler.get_last_lr()[0]}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_logits, all_labels = [], []

        for batch in self.val_loader:
            cart_emb = batch["cart_embeddings"].to(self.device)
            target_emb = batch["target_embedding"].to(self.device)
            mask = batch["padding_mask"].to(self.device)
            temporal = batch["temporal_features"].to(self.device)
            geohash = batch["geohash_bucket"].to(self.device)
            labels = batch["label"].to(self.device)

            output = self.model(cart_embeddings=cart_emb, target_embeddings=target_emb,
                                padding_mask=mask, temporal_features=temporal,
                                geohash_buckets=geohash, labels=labels)
            total_loss += output["loss"].item()
            num_batches += 1
            all_logits.extend(output["logits"].cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

        avg_loss = total_loss / max(num_batches, 1)
        auc = self.metrics.compute_auc(np.array(all_logits), np.array(all_labels))
        return {"val_loss": avg_loss, "val_auc": auc}

    def train(self) -> Dict[str, Any]:
        print_banner("Training SwaadStack Model")
        console.print(f"  Device: {self.device}")
        console.print(f"  Epochs: {self.config.num_epochs}")
        console.print(f"  Batch size: {self.config.batch_size}")
        console.print(f"  Train batches: {len(self.train_loader)}")
        console.print(f"  Val batches: {len(self.val_loader)}")

        best_metrics = {}
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            epoch_time = time.time() - start_time
            metrics = {**train_metrics, **val_metrics, "epoch_time": epoch_time}
            self.training_history.append(metrics)

            console.print(
                f"\n📈 Epoch {epoch+1}: Train Loss={metrics['train_loss']:.4f}, "
                f"Val Loss={metrics['val_loss']:.4f}, Train AUC={metrics['train_auc']:.4f}, "
                f"Val AUC={metrics['val_auc']:.4f}, Time={epoch_time:.1f}s"
            )

            if val_metrics["val_loss"] < self.best_val_loss - self.config.min_delta:
                self.best_val_loss = val_metrics["val_loss"]
                self.patience_counter = 0
                best_metrics = metrics.copy()
                self._save_checkpoint(epoch, is_best=True)
                console.print("  💾 [green]New best model saved![/green]")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    console.print(f"\n⏹ [yellow]Early stopping at epoch {epoch+1}[/yellow]")
                    break

        console.print("\n[bold green]✅ Training Complete![/bold green]")
        if best_metrics:
            print_metrics(best_metrics, "Best Model Metrics")
        return {"history": self.training_history, "best_metrics": best_metrics, "best_val_loss": self.best_val_loss}

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        path = model_config.model_weights_path
        torch.save(self.model.state_dict(), str(path))
        if is_best:
            best_path = str(path).replace(".pth", "_best.pth")
            torch.save(self.model.state_dict(), best_path)
