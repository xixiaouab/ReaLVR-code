"""Stage-1 trainer: next-token loss plus the weighted visual reconstruction loss.

Adapted from LVR (https://github.com/VincentLeebang/lvr, Apache-2.0) with modifications.
"""

import math
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainerCallback


class DisableFrozenVisionCheckpointing(TrainerCallback):
    """Turns off activation checkpointing in a frozen vision tower, which runs without gradients."""

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if any(p.requires_grad for p in model.visual.parameters()):
            return
        for module in model.visual.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = False


class LVRSFTTrainer(Trainer):
    """Trains on packed batches with loss = loss_ce + loss_lvr_lambda * loss_lvr and logs both terms."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # compute_loss returns a per-batch mean, so the Trainer has to scale it for gradient accumulation.
        self.model_accepts_loss_kwargs = False
        self.add_callback(DisableFrozenVisionCheckpointing)
        self._metrics = defaultdict(list)

    def get_train_dataloader(self):
        """The packed dataset gives every rank and worker its own shard, so the loader is not sharded again."""
        return DataLoader(
            self.train_dataset,
            batch_size=self._train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
            prefetch_factor=self.args.dataloader_prefetch_factor,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss_ce
        if outputs.loss_lvr is not None:
            loss = loss + self.args.loss_lvr_lambda * outputs.loss_lvr
        self._record_losses(outputs.loss_ce, outputs.loss_lvr)
        return (loss, outputs) if return_outputs else loss

    def _record_losses(self, loss_ce, loss_lvr):
        """Average both terms over processes; a batch without <|lvr|> tokens has no reconstruction loss."""
        if loss_lvr is None:
            loss_lvr = torch.full_like(loss_ce, float("nan"))
        values = torch.stack([loss_ce.detach().float(), loss_lvr.detach().float()])
        values = self.accelerator.gather(values[None]).nanmean(dim=0)
        for name, value in zip(("loss_ce", "loss_lvr"), values.tolist()):
            if not math.isnan(value):
                self._metrics[name].append(value)

    def log(self, logs, start_time=None):
        if "loss" in logs and self._metrics:
            logs.update({name: sum(values) / len(values) for name, values in self._metrics.items()})
            self._metrics.clear()
        super().log(logs, start_time)
