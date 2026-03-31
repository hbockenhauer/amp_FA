import os
import numpy as np
from src.model.utils import LiDARInstance3DBoxes

import torch
import torchvision.transforms as T
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
                 use_painted_radar=False,
                 painted_radar_dir='painted_radar',
                 radar_mode='single',
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
        assert split in ['train', 'val', 'test'], (
            f"Invalid split: {split}. Must be one of ['train', 'val', 'test']")
        assert radar_mode in self.RADAR_MODES, (
            f"radar_mode must be one of {list(self.RADAR_MODES.keys())}")

        self.split = split
        self.use_painted_radar = use_painted_radar
        self.painted_radar_dir = painted_radar_dir
        self.radar_mode = radar_mode

        self.vod_kitti_locations = KittiLocations(root_dir=data_root)
        radar_folder = self.RADAR_MODES[radar_mode]
        self.vod_kitti_locations.radar_dir = os.path.join(
            data_root, radar_folder, 'training', 'velodyne')

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
        
        self.vod_kitti_locations = KittiLocations(root_dir=data_root)

        # Determine model type from loaded config, applicable for pointPainting method inferencing
        self.painting_model_type = None
        if self.use_painted_radar:
            if 'resnet' in self.painted_radar_dir.lower():
                self.painting_model_type = 'resnet'
            elif 'mobilenet' in self.painted_radar_dir.lower():
                self.painting_model_type = 'mobilenet'
                
            self.seg_model = None # We will load this only when a file is missing
            self.image_transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])


 
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
    
    def _paint_radar_inference(self, frame_data, transforms, raw_radar_data):
        # Process unseen data for pointPainting model (Used for inferencing or test set)
        
        # Load the model only once to save time
        if self.seg_model is None:
            if self.painting_model_type == 'resnet':
                from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
                self.seg_model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT).cuda().eval()

            elif self.painting_model_type == 'mobilenet':
                from torchvision.models.segmentation import lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights
                self.seg_model = lraspp_mobilenet_v3_large(weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT).cuda().eval()

            else:
                raise ValueError("Could not find model type in folder name!")
        
        image = frame_data.image
        
        # Segment Image
        img_tensor = self.image_transform(image).unsqueeze(0).cuda()
        with torch.no_grad():
            output = self.seg_model(img_tensor)['out'][0]
            seg_probs = torch.softmax(output, dim=0).cpu().numpy()
            
        # Project Radar Points
        trans_homo_radar = np.ones((raw_radar_data.shape[0], 4))
        trans_homo_radar[:, :3] = raw_radar_data[:, :3]
        t_camProj_lidar = transforms.camera_projection_matrix @ transforms.t_camera_lidar
        uv_coords_homo = (t_camProj_lidar @ trans_homo_radar.T).T
        
        u = (uv_coords_homo[:, 0] / uv_coords_homo[:, 2]).astype(int)
        v = (uv_coords_homo[:, 1] / uv_coords_homo[:, 2]).astype(int)
        
        H, W = image.shape[:2]
        valid_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (uv_coords_homo[:, 2] > 0)
        
        # Extract correct channels: Car(7), Person(15), Bicycle(2)
        painted_channels = np.zeros((raw_radar_data.shape[0], 3))
        painted_channels[valid_mask, 0] = seg_probs[7, v[valid_mask], u[valid_mask]]
        painted_channels[valid_mask, 1] = seg_probs[15, v[valid_mask], u[valid_mask]]
        painted_channels[valid_mask, 2] = seg_probs[2, v[valid_mask], u[valid_mask]]
        
        # Combine original radar with the 3 new probability channels
        painted_radar = np.concatenate([raw_radar_data, painted_channels], axis=1)
        return painted_radar

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        num_frame = self.sample_list[idx]
        vod_frame_data = FrameDataLoader(kitti_locations=self.vod_kitti_locations, frame_number=num_frame)
        local_transforms = FrameTransformMatrix(vod_frame_data)
        
        #########################################################################################################
        # Check if pre-processed files exist, otherwise do live processing (much slower)
        if self.use_painted_radar:
            npy_path = os.path.join(os.getcwd(), self.painted_radar_dir, f"{num_frame}.npy")
            
            if os.path.exists(npy_path):
                # File exists, load it quickly
                radar_data = np.load(npy_path)
            else:
                # File is missing (hidden test set), load raw data and paint it live
                raw_radar = vod_frame_data.radar_data
                radar_data = self._paint_radar_inference(vod_frame_data, local_transforms, raw_radar)
        else:
            # Baseline model (no painting, raw radar points)
            radar_data = vod_frame_data.radar_data
        #######################################################################################################

        radar_data = self._apply_doppler_preprocessing(radar_data)
        radar_data = self._apply_training_augmentation(radar_data)
        gt_labels_3d_list = []
        gt_bboxes_3d_list = []
        if self.split != 'test':
            raw_labels = vod_frame_data.raw_labels
            for idx_lbl, label in enumerate(raw_labels):
                label = label.split(' ')
                
                if label[self.LABEL_MAPPING['class']] in self.CLASSES: 
                    gt_labels_3d_list.append(int(self.CLASSES.index(label[self.LABEL_MAPPING['class']])))

                    bbox3d_loc_camera = np.array(label[self.LABEL_MAPPING['bbox3d_location']])
                    trans_homo_cam = np.ones((1,4))
                    trans_homo_cam[:, :3] = bbox3d_loc_camera
                    bbox3d_loc_lidar = homogeneous_transformation(trans_homo_cam, local_transforms.t_lidar_camera)
                    
                    bbox3d_locs = np.array(bbox3d_loc_lidar[0,:3], dtype=np.float32)         
                    bbox3d_dims = np.array(label[self.LABEL_MAPPING['bbox3d_dimensions']], dtype=np.float32)[[2, 1, 0]] 
                    bbox3d_rot = np.array([label[self.LABEL_MAPPING['bbox3d_rotation']]], dtype=np.float32)
                
                    gt_bboxes_3d_list.append(np.concatenate([bbox3d_locs, bbox3d_dims, bbox3d_rot], axis=0))

        radar_data = torch.tensor(radar_data, dtype=torch.float32)
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
            meta=dict(
                num_frame=num_frame 
            )
        )