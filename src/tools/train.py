import os
import sys
import os.path as osp
import threading
root = os.path.abspath(os.path.join(os.getcwd()))
if root not in sys.path:
    sys.path.insert(0, root)
    
import hydra
import wandb
from  omegaconf import DictConfig, OmegaConf

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

import torch
from torch.utils.data import DataLoader
from src.model.detector import CenterPoint
from src.dataset import ViewOfDelft, collate_vod_batch


def _finish_wandb_with_timeout(timeout_s: float = 20.0) -> bool:
    """Finalize W&B without allowing teardown to block SLURM slot release.

    Returns:
        bool: True if W&B teardown appears to be hanging after timeout.
    """
    if wandb.run is None:
        return False

    errors = []

    def _finish() -> None:
        try:
            wandb.finish(exit_code=0)
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=_finish, daemon=True)
    thread.start()
    thread.join(timeout=timeout_s)

    is_hanging = False
    if thread.is_alive():
        is_hanging = True
        print(
            f"Warning: wandb.finish() exceeded {timeout_s:.0f}s, forcing teardown.",
            flush=True,
        )
        try:
            wandb.teardown()
        except Exception as exc:
            print(f"Warning: wandb.teardown() raised exception: {exc}", flush=True)

    if errors:
        print(f"Warning: wandb.finish() raised exception: {errors[0]}", flush=True)

    return is_hanging

@hydra.main(version_base=None, config_path='../config', config_name='train')    
def train(cfg: DictConfig)-> None:
    L.seed_everything(cfg.seed, workers=True)

    dataset_cfg = {}
    if 'dataset' in cfg.model:
        dataset_cfg = OmegaConf.to_container(cfg.model.dataset, resolve=True)
    
    train_dataset = ViewOfDelft(
        data_root=cfg.data_root,
        split='train',
        **dataset_cfg)
    val_dataset = ViewOfDelft(
        data_root=cfg.data_root,
        split='val',
        **dataset_cfg)
    
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=cfg.batch_size, 
                                  num_workers=cfg.num_workers, 
                                  shuffle=True,
                                  collate_fn=collate_vod_batch)
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=1, 
                                num_workers=cfg.num_workers, 
                                shuffle=False,
                                collate_fn=collate_vod_batch)
    model = CenterPoint(cfg.model)
    callbacks = [
        ModelCheckpoint(
            dirpath=osp.join(cfg.output_dir, "checkpoints"),
            filename='ep{epoch}-'+cfg.exp_id,
            save_last=True,
            monitor='validation/entire_area/mAP',
            mode='max',
            auto_insert_metric_name=False,
            save_top_k=cfg.save_top_model,
        ),
        LearningRateMonitor(logging_interval="epoch")
    ]

    early_cfg = cfg.get('early_stopping', {})
    if early_cfg.get('enabled', False):
        callbacks.append(
            EarlyStopping(
                monitor=early_cfg.get('monitor', 'validation/entire_area/mAP'),
                mode=early_cfg.get('mode', 'max'),
                patience=early_cfg.get('patience', 2),
                min_delta=early_cfg.get('min_delta', 0.0),
                strict=early_cfg.get('strict', True),
                verbose=True,
            ))

    logger = WandbLogger(
        save_dir=osp.join(cfg.output_dir, 'wandb_logs'),
        project='amp',
        name=cfg.exp_id,
        log_model=False,
        mode='online',
    )
    logger.watch(model, log_graph=False)
        

    
    trainer = L.Trainer(
        logger=logger,
        log_every_n_steps=cfg.log_every,
        accelerator="gpu",
        devices=cfg.gpus,
        check_val_every_n_epoch=cfg.val_every,
        strategy="auto",
        callbacks=callbacks,
        max_epochs=cfg.epochs,
        sync_batchnorm=cfg.sync_bn,
        enable_model_summary= True,
    )

    try:
        trainer.fit(model,
                    train_dataloaders=train_dataloader,
                    val_dataloaders=val_dataloader,
                    ckpt_path=cfg.checkpoint_path)
    finally:
        timeout_s = float(cfg.get('wandb_finish_timeout_s', 20.0))
        force_exit_on_hang = bool(cfg.get('force_exit_on_wandb_hang', True))
        hanging = _finish_wandb_with_timeout(timeout_s=timeout_s)
        # As a last resort, force-exit so SLURM can release the slot and start queued jobs.
        if hanging and force_exit_on_hang:
            print("Warning: forcing process exit after W&B hang to unblock scheduler.", flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
    
if __name__ == '__main__':
    train()