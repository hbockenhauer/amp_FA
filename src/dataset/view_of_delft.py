import os
import numpy as np
from src.model.utils import LiDARInstance3DBoxes

import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
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
    
    def __init__(self, data_root='data/view_of_delft', sequential_loading=False, split='train'):
        super().__init__()
        
        self.data_root = data_root
        assert split in ['train', 'val', 'test'], f"Invalid split: {split}. Must be one of ['train', 'val', 'test']"
        self.split = split
        split_file = os.path.join(data_root, 'lidar', 'ImageSets', f'{split}.txt')

        with open(split_file, 'r') as f:
            lines = f.readlines()
            self.sample_list = [line.strip() for line in lines]
        
        self.vod_kitti_locations = KittiLocations(root_dir=data_root)

        # 1. Initialize the semantic segmentation model (DeepLabV3)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seg_model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        self.seg_model.to(self.device).eval()
        self.image_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        num_frame = self.sample_list[idx]
        vod_frame_data = FrameDataLoader(kitti_locations=self.vod_kitti_locations, frame_number=num_frame)
        local_transforms = FrameTransformMatrix(vod_frame_data)
        
        radar_data = vod_frame_data.radar_data

        # --- NEW FAST LOADING --- (Requires dataset preprocessing with: preprocess_pointpainting.py)
        folder_name = 'painted_radar'
        painted_radar_path = os.path.join(os.getcwd(), folder_name, f'{num_frame}.npy')
        radar_data = np.load(painted_radar_path)
        radar_data = torch.tensor(radar_data, dtype=torch.float32)
        # ------------------------

        ## --- POINT PAINTING IMPLEMENTATION ---
        #image = vod_frame_data.image 
        #
        #with torch.no_grad():
        #    img_tensor = self.image_transform(image).unsqueeze(0).to(self.device)
        #    seg_output = self.seg_model(img_tensor)['out'][0] 
        #    seg_probs = torch.softmax(seg_output, dim=0) 
        #    painted_channels = seg_probs[[3, 1, 2], :, :] 
        #
        #trans_homo_radar = np.ones((radar_data.shape[0], 4))
        #trans_homo_radar[:, :3] = radar_data[:, :3]
#
        #t_camProj_lidar = local_transforms.camera_projection_matrix @ local_transforms.t_camera_lidar
#
        ##uv_coords_homo = homogeneous_transformation(trans_homo_radar, t_camProj_lidar)
        #uv_coords_homo = (t_camProj_lidar @ trans_homo_radar.T).T
        #u = (uv_coords_homo[:, 0] / uv_coords_homo[:, 2]).astype(int)
        #v = (uv_coords_homo[:, 1] / uv_coords_homo[:, 2]).astype(int)
        #
        #H, W = image.shape[:2]
        #valid_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (uv_coords_homo[:, 2] > 0)
        #
        #point_scores = np.zeros((radar_data.shape[0], 3), dtype=np.float32)
        #valid_u = u[valid_mask]
        #valid_v = v[valid_mask]
        #point_scores[valid_mask] = painted_channels[:, valid_v, valid_u].cpu().numpy().T
        #
        #radar_data = np.concatenate([radar_data, point_scores], axis=-1)
        ## -------------------------------------

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
            lidar_data=radar_data,
            gt_labels_3d=gt_labels_3d,
            gt_bboxes_3d=gt_bboxes_3d,
            meta=dict(
                num_frame=num_frame 
            )
        )