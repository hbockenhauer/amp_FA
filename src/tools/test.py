import os
import sys
root = os.path.abspath(os.path.join(os.getcwd()))
if root not in sys.path:
    sys.path.insert(0, root)
    
import hydra
from  omegaconf import DictConfig, OmegaConf

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

import torch
from torch.utils.data import DataLoader

from src.model.detector import CenterPoint
from src.dataset import ViewOfDelft, collate_vod_batch


@hydra.main(version_base=None, config_path='../config', config_name="test")
def eval(cfg: DictConfig) -> None:
    print('Evaluating model...')
    L.seed_everything(cfg.seed, workers=True)

    checkpoint = torch.load(cfg.checkpoint_path,weights_only=False)
    checkpoint_params = DictConfig(checkpoint["hyper_parameters"])
    print('checkpoint params:', checkpoint_params.keys())
    print('checkpoint params cfg:', checkpoint_params.config.keys())

    model_cfg = checkpoint_params.config.model if 'config' in checkpoint_params else cfg.model
    dataset_kwargs = dict(
        radar_mode=model_cfg.get('radar_mode', 'single'),
        motion_compensation=model_cfg.get('motion_compensation', False),
        motion_dt=model_cfg.get('motion_dt', 0.1),
        vr_channel_idx=model_cfg.get('vr_channel_idx', 4),
        time_channel_idx=model_cfg.get('time_channel_idx', 6),
    )

    test_dataset = ViewOfDelft(data_root=cfg.data_root, split='test', **dataset_kwargs)
    test_dataloader = DataLoader(test_dataset,
                                batch_size=1,
                                num_workers=cfg.num_workers,
                                shuffle=False,
                                collate_fn=collate_vod_batch)
    
    model = CenterPoint.load_from_checkpoint(checkpoint_path=cfg.checkpoint_path)
    model.inference_mode = 'test'
    model.save_results = True
    model.eval()
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
    )
    
    trainer.validate(model = model, dataloaders=test_dataloader)
                     
    
if __name__ == '__main__':
    eval()