import numpy as np
import torch
from pytorch_lightning.callbacks import Callback
import wandb
import tqdm

class SmoothedLossMonitor(Callback):
    def __init__(self, window_size=5):
        super().__init__()
        self.window_size = window_size
        self.history = []

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get('val_loss')
        if val_loss is not None:
            val_loss = val_loss.item()
            self.history.append(val_loss)
            if len(self.history) > self.window_size:
                self.history.pop(0)
            smoothed_loss = np.mean(self.history)
            trainer.callback_metrics['smoothed_val_loss'] = torch.tensor(smoothed_loss)


class GapEarlyStopping(Callback):
    def __init__(self, threshold=0.1, patience=5):
        super().__init__()
        self.threshold = threshold
        self.patience = patience
        self.wait = 0

    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        val_loss = metrics.get("val_loss")
        train_loss = metrics.get("train_loss")

        if val_loss is not None and train_loss is not None:
            gap = val_loss - train_loss
            tqdm.write(f"[EarlyStopping] Val-Train gap: {gap:.4f}")

            if gap > self.threshold:
                self.wait += 1
                if self.wait >= self.patience:
                    trainer.should_stop = True
            else:
                self.wait = 0


class SaveWandbInfoCallback(Callback):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if wandb.run:
            checkpoint["wandb_run_id"] = wandb.run.id
            checkpoint["wandb_run_name"] = wandb.run.name


class LogGradNormCallback(Callback):
    def __init__(self, log_every_n_steps=50):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_after_backward(self, trainer, pl_module):
        # Calcula la norma L2 de todos los gradientes
        total_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
        total_norm = total_norm ** 0.5

        # Loggea en wandb cada n steps
        global_step = trainer.global_step
        if global_step % self.log_every_n_steps == 0:
            trainer.logger.experiment.log(
                {"grad_norm": total_norm, "global_step": global_step}
            )