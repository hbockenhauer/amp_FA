import os
import numpy as np
from src.model.utils import LiDARInstance3DBoxes

import torch
from torch.utils.data import Dataset

from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader, FrameTransformMatrix, homogeneous_transformation

class ViewOfDelft(Dataset):
    CLASSES = ['Car', 'Pedestrian', 'Cyclist']
    
    LABEL_MAPPING = {
        'class': 0,
        'truncated': 1,
        'occluded': 2,
        'alpha': 3,
        'bbox2d': slice(4,8),
        'bbox3d_dimensions': slice(8,11),
        'bbox3d_location': slice(11,14),
        'bbox3d_rotation': 14,
    }

    RADAR_MODES = {
        'single':   'radar',
        '3_frames': 'radar_3frames',
        '5_frames': 'radar_5frames',
    }

    def __init__(self,
                 data_root='data/view_of_delft',
                 sequential_loading=False,
                 split='train',
                 radar_mode='single',
                 motion_compensation=False,
                 motion_dt=0.1,
                 vr_channel_idx=5,
                 time_channel_idx=6):
        super().__init__()
        
        self.data_root = data_root
        assert split in ['train', 'val', 'test']
        assert radar_mode in self.RADAR_MODES, \
            f"radar_mode must be one of {list(self.RADAR_MODES.keys())}"
        
        self.split = split
        self.radar_mode = radar_mode
        self.motion_compensation = motion_compensation
        self.motion_dt = float(motion_dt)
        self.vr_channel_idx = int(vr_channel_idx)
        self.time_channel_idx = int(time_channel_idx)

        # root_dir stays at dataset root so labels/camera/pose all resolve correctly
        self.vod_kitti_locations = KittiLocations(root_dir=data_root)
        
        # Override only radar_dir to point at the correct scan folder
        radar_folder = self.RADAR_MODES[radar_mode]
        self.vod_kitti_locations.radar_dir = os.path.join(
            data_root, radar_folder, 'training', 'velodyne')
        # radar_calib_dir stays pointing at radar/training/calib — same sensor, same calib

        split_file = os.path.join(data_root, 'lidar', 'ImageSets', f'{split}.txt')
        with open(split_file, 'r') as f:
            self.sample_list = [line.strip() for line in f.readlines()]

    def _compensate_temporal_points(self, radar_data: np.ndarray) -> np.ndarray:
        """Compensate past-frame points toward the current frame using radial velocity.
        """
        if (not self.motion_compensation) or self.radar_mode == 'single':
            return radar_data
        if radar_data.ndim != 2:
            return radar_data

        num_channels = radar_data.shape[1]
        if num_channels <= max(self.vr_channel_idx, self.time_channel_idx):
            return radar_data

        out = radar_data.copy()
        x = out[:, 0]
        y = out[:, 1]
        vr = out[:, self.vr_channel_idx]
        frame_offset = out[:, self.time_channel_idx]

        # Temporal bins store 0 for current points and negative integers for past frames.
        dt_seconds = np.maximum(-frame_offset, 0.0) * self.motion_dt

        r = np.sqrt(x * x + y * y)
        unit_x = np.zeros_like(x)
        unit_y = np.zeros_like(y)
        valid = r > 1e-6
        unit_x[valid] = x[valid] / r[valid]
        unit_y[valid] = y[valid] / r[valid]

        vx = vr * unit_x
        vy = vr * unit_y
        out[:, 0] = x + vx * dt_seconds
        out[:, 1] = y + vy * dt_seconds

        return out

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        num_frame = self.sample_list[idx]
        vod_frame_data = FrameDataLoader(
            kitti_locations=self.vod_kitti_locations,
            frame_number=num_frame)
        local_transforms = FrameTransformMatrix(vod_frame_data)

        radar_data = vod_frame_data.radar_data  # (N, 7) for all modes
        radar_data = self._compensate_temporal_points(radar_data)

        gt_labels_3d_list = []
        gt_bboxes_3d_list = []
        if self.split != 'test':
            raw_labels = vod_frame_data.raw_labels
            for _, label in enumerate(raw_labels):
                label = label.split(' ')
                if label[self.LABEL_MAPPING['class']] in self.CLASSES:
                    gt_labels_3d_list.append(
                        int(self.CLASSES.index(label[self.LABEL_MAPPING['class']])))
                    bbox3d_loc_camera = np.array(
                        label[self.LABEL_MAPPING['bbox3d_location']])
                    trans_homo_cam = np.ones((1, 4))
                    trans_homo_cam[:, :3] = bbox3d_loc_camera
                    bbox3d_loc_lidar = homogeneous_transformation(
                        trans_homo_cam, local_transforms.t_lidar_camera)
                    bbox3d_locs = np.array(
                        bbox3d_loc_lidar[0, :3], dtype=np.float32)
                    bbox3d_dims = np.array(
                        label[self.LABEL_MAPPING['bbox3d_dimensions']],
                        dtype=np.float32)[[2, 1, 0]]
                    bbox3d_rot = np.array(
                        [label[self.LABEL_MAPPING['bbox3d_rotation']]],
                        dtype=np.float32)
                    gt_bboxes_3d_list.append(
                        np.concatenate([bbox3d_locs, bbox3d_dims, bbox3d_rot], axis=0))

        radar_data = torch.tensor(radar_data)

        if gt_bboxes_3d_list == []:
            gt_labels_3d = np.array([0])
            gt_bboxes_3d = np.zeros((1, 7))
        else:
            gt_labels_3d = np.array(gt_labels_3d_list, dtype=np.int64)
            gt_bboxes_3d = np.stack(gt_bboxes_3d_list, axis=0)

        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0))
        gt_labels_3d = torch.tensor(gt_labels_3d)

        return dict(
            lidar_data=radar_data,
            gt_labels_3d=gt_labels_3d,
            gt_bboxes_3d=gt_bboxes_3d,
            meta=dict(num_frame=num_frame)
        )