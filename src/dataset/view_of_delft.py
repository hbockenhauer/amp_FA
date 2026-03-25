import os
import numpy as np
from src.model.utils import LiDARInstance3DBoxes

import torch
from torch.utils.data import Dataset

from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader, FrameTransformMatrix, homogeneous_transformation

class ViewOfDelft(Dataset):
    CLASSES = ['Car', 
               'Pedestrian', 
               'Cyclist',]
            #    'rider', 
            #    'unused_bicycle', 
            #    'bicycle_rack', 
            #    'human_depiction', 
            #    'moped_or_scooter', 
            #    'motor',
            #    'truck',
            #    'other_ride',
            #    'other_vehicle',
            #    'uncertain_ride'
    
    LABEL_MAPPING = {
        'class': 0, # Describes the type of object: 'Car', 'Pedestrian', 'Cyclist', etc.
        'truncated': 1, # Not used, only there to be compatible with KITTI format.
        'occluded': 2, # Integer (0,1,2) indicating occlusion state 0 = fully visible, 1 = partly occluded 2 = largely occluded.
        'alpha': 3, # Observation angle of object, ranging [-pi..pi]
        'bbox2d': slice(4,8),
        'bbox3d_dimensions': slice(8,11), # 3D object dimensions: height, width, length (in meters).
        'bbox3d_location': slice(11,14), # 3D object location x,y,z in camera coordinates (in meters).
        'bbox3d_rotation': 14, # Rotation around -Z-axis in LiDAR coordinates [-pi..pi].
    }
    
    def __init__(self, 
                 data_root = 'data/view_of_delft', 
                 sequential_loading=False,
                 split = 'train',
                 doppler_index=5,
                 rcs_index=3,
                 doppler_clip=None,
                 doppler_normalize=False,
                 drop_static_points=False,
                 static_velocity_threshold=0.2,
                 static_keep_prob=1.0,
                 augment_train_points=False,
                 point_dropout_prob=0.0,
                 xy_noise_std=0.0,
                 rcs_noise_std=0.0,
                 doppler_noise_std=0.0):
        super().__init__()
        
        self.data_root = data_root
        assert split in ['train', 'val', 'test'], f"Invalid split: {split}. Must be one of ['train', 'val', 'test']"
        self.split = split
        split_file = os.path.join(data_root, 'lidar', 'ImageSets', f'{split}.txt')

        with open(split_file, 'r') as f:
            lines = f.readlines()
            self.sample_list = [line.strip() for line in lines]

        self.doppler_index = doppler_index
        self.rcs_index = rcs_index
        self.doppler_clip = doppler_clip
        self.doppler_normalize = doppler_normalize
        self.drop_static_points = drop_static_points
        self.static_velocity_threshold = static_velocity_threshold
        self.static_keep_prob = static_keep_prob
        self.augment_train_points = augment_train_points
        self.point_dropout_prob = point_dropout_prob
        self.xy_noise_std = xy_noise_std
        self.rcs_noise_std = rcs_noise_std
        self.doppler_noise_std = doppler_noise_std
        
        self.vod_kitti_locations = KittiLocations(root_dir = data_root)

    def _apply_doppler_preprocessing(self, radar_data):
        """Apply optional Doppler preprocessing for ablation studies."""
        if radar_data is None:
            return radar_data

        arr = np.asarray(radar_data)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return arr
        if self.doppler_index < 0 or self.doppler_index >= arr.shape[1]:
            return arr

        out = arr.copy()
        doppler = out[:, self.doppler_index]

        if self.doppler_clip is not None and len(self.doppler_clip) == 2:
            dmin, dmax = self.doppler_clip
            doppler = np.clip(doppler, dmin, dmax)

        if self.doppler_normalize:
            mu = float(np.mean(doppler))
            sigma = float(np.std(doppler))
            doppler = (doppler - mu) / max(sigma, 1e-6)

        out[:, self.doppler_index] = doppler

        if self.drop_static_points:
            moving_mask = np.abs(doppler) >= self.static_velocity_threshold
            keep_mask = moving_mask
            if self.split == 'train' and self.static_keep_prob < 1.0:
                static_mask = ~moving_mask
                keep_static = np.random.rand(static_mask.sum()) < self.static_keep_prob
                keep_mask = moving_mask.copy()
                keep_mask[static_mask] = keep_static

            if keep_mask.any():
                out = out[keep_mask]

        return out

    def _apply_training_augmentation(self, radar_data):
        """Apply optional point-level augmentation without label transforms."""
        arr = np.asarray(radar_data)
        if (self.split != 'train' or not self.augment_train_points or
                arr.ndim != 2 or arr.shape[0] == 0):
            return arr

        out = arr.copy()

        if self.point_dropout_prob > 0.0:
            keep = np.random.rand(out.shape[0]) > self.point_dropout_prob
            if keep.any():
                out = out[keep]

        if self.xy_noise_std > 0.0 and out.shape[1] >= 2:
            out[:, 0] += np.random.normal(0.0, self.xy_noise_std, size=out.shape[0])
            out[:, 1] += np.random.normal(0.0, self.xy_noise_std, size=out.shape[0])

        if self.rcs_noise_std > 0.0 and 0 <= self.rcs_index < out.shape[1]:
            out[:, self.rcs_index] += np.random.normal(
                0.0, self.rcs_noise_std, size=out.shape[0])

        if self.doppler_noise_std > 0.0 and 0 <= self.doppler_index < out.shape[1]:
            out[:, self.doppler_index] += np.random.normal(
                0.0, self.doppler_noise_std, size=out.shape[0])

        return out
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        num_frame = self.sample_list[idx]
        vod_frame_data = FrameDataLoader(kitti_locations=self.vod_kitti_locations,
                                         frame_number=num_frame)
        local_transforms = FrameTransformMatrix(vod_frame_data)
        
        radar_data = vod_frame_data.radar_data
        radar_data = self._apply_doppler_preprocessing(radar_data)
        radar_data = self._apply_training_augmentation(radar_data)

        
        gt_labels_3d_list = []
        gt_bboxes_3d_list = []
        if self.split != 'test':
            raw_labels = vod_frame_data.raw_labels
            for idx, label in enumerate(raw_labels):
                label = label.split(' ')
                
                if label[self.LABEL_MAPPING['class']] in self.CLASSES: 

                    gt_labels_3d_list.append(int(self.CLASSES.index(label[self.LABEL_MAPPING['class']])))

                    bbox3d_loc_camera = np.array(label[self.LABEL_MAPPING['bbox3d_location']])
                    trans_homo_cam = np.ones((1,4))
                    trans_homo_cam[:, :3] = bbox3d_loc_camera
                    bbox3d_loc_lidar = homogeneous_transformation(trans_homo_cam, local_transforms.t_lidar_camera)
                    
                    bbox3d_locs = np.array(bbox3d_loc_lidar[0,:3], dtype=np.float32)         
                    bbox3d_dims = np.array(label[self.LABEL_MAPPING['bbox3d_dimensions']], dtype=np.float32)[[2, 1, 0]] # hwl -> lwh
                    bbox3d_rot = np.array([label[self.LABEL_MAPPING['bbox3d_rotation']]], dtype=np.float32)
                
                    gt_bboxes_3d_list.append(np.concatenate([bbox3d_locs, bbox3d_dims, bbox3d_rot], axis=0))

        radar_data = torch.tensor(radar_data)
        
        if gt_bboxes_3d_list == []:
            gt_labels_3d = np.array([0])
            gt_bboxes_3d = np.zeros((1,7))
        else:
            gt_labels_3d = np.array(gt_labels_3d_list, dtype=np.int64)
            gt_bboxes_3d = np.stack(gt_bboxes_3d_list, axis=0)
        
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0))
        
        gt_labels_3d = torch.tensor(gt_labels_3d)
        
        return dict(
            lidar_data = radar_data,
            gt_labels_3d = gt_labels_3d,
            gt_bboxes_3d = gt_bboxes_3d,
            meta = dict(
                num_frame = num_frame 
            )
        )
    

        