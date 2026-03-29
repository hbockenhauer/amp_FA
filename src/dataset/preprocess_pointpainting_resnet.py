import os
import sys
import time
import argparse
import numpy as np

# Add project root to path
root = os.path.abspath(os.path.join(os.getcwd()))
if root not in sys.path:
    sys.path.insert(0, root)

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader, FrameTransformMatrix

# 1. Custom Dataset to load files using background workers
class VoDPreprocessDataset(Dataset):
    RADAR_MODES = {
        'single': 'radar',
        '3_frames': 'radar_3frames',
        '5_frames': 'radar_5frames',
    }

    def __init__(self, data_root, sample_list, radar_mode='single'):
        self.sample_list = sample_list
        if radar_mode not in self.RADAR_MODES:
            raise ValueError(
                f"Unsupported radar_mode={radar_mode}. Choose from {list(self.RADAR_MODES.keys())}")

        self.vod_kitti_locations = KittiLocations(root_dir=data_root)
        radar_folder = self.RADAR_MODES[radar_mode]
        self.vod_kitti_locations.radar_dir = os.path.join(
            data_root, radar_folder, 'training', 'velodyne')

        self.image_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        num_frame = self.sample_list[idx]
        vod_frame = FrameDataLoader(kitti_locations=self.vod_kitti_locations, frame_number=num_frame)
        local_transforms = FrameTransformMatrix(vod_frame)
        
        # .copy() fixes the PyTorch writable warning
        img_tensor = self.image_transform(vod_frame.image.copy())
        
        return {
            'image': img_tensor,
            'radar': vod_frame.radar_data,
            't_camProj': local_transforms.camera_projection_matrix @ local_transforms.t_camera_lidar,
            'frame_num': num_frame
        }

# 2. Function to group different sized radar arrays into batches
def custom_collate(batch):
    images = torch.stack([item['image'] for item in batch])
    radars = [item['radar'] for item in batch]
    t_camProjs = [item['t_camProj'] for item in batch]
    frame_nums = [item['frame_num'] for item in batch]
    return images, radars, t_camProjs, frame_nums

def parse_args():
    parser = argparse.ArgumentParser(description='Precompute ResNet pointpainting radar features.')
    parser.add_argument('--data-root', default='data/view_of_delft',
                        help='Root directory of the View-of-Delft dataset.')
    parser.add_argument('--radar-mode', default='single', choices=['single', '3_frames', '5_frames'],
                        help='Radar source to paint.')
    parser.add_argument('--save-dir', default=None,
                        help='Output directory for painted radar .npy files.')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size for segmentation inference.')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='DataLoader workers.')
    parser.add_argument('--prefetch-factor', type=int, default=1,
                        help='DataLoader prefetch factor.')
    return parser.parse_args()


def preprocess_dataset(args):
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 3. Load the fast ResNet model
    seg_model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT).to(device).eval()
    
    data_root = args.data_root
    default_save_dirs = {
        'single': 'painted_radar_resnet',
        '3_frames': 'painted_radar_resnet_3frames',
        '5_frames': 'painted_radar_resnet_5frames',
    }
    save_dir_name = args.save_dir if args.save_dir else default_save_dirs[args.radar_mode]
    save_dir = os.path.join(os.getcwd(), save_dir_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Radar mode: {args.radar_mode}")
    print(f"Save directory: {save_dir}")
    
    all_frames = []
    for split in ['train', 'val', 'test']:
        split_file = os.path.join(data_root, 'lidar', 'ImageSets', f'{split}.txt')
        with open(split_file, 'r') as f:
            all_frames.extend([line.strip() for line in f.readlines()])
            
    # Skip files you already processed in the slow run
    frames_to_process = [f for f in all_frames if not os.path.exists(os.path.join(save_dir, f'{f}.npy'))]
    print(f"Frames left to process: {len(frames_to_process)}")
    
    dataset = VoDPreprocessDataset(data_root, frames_to_process, radar_mode=args.radar_mode)
    
    # Multiprocessing and Batching
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=custom_collate,
        prefetch_factor=args.prefetch_factor)
    
    for batch_idx, (images, radars, t_camProjs, frame_nums) in enumerate(dataloader):
        images = images.to(device)
        
        # Process 16 images at once
        with torch.no_grad():
            outputs = seg_model(images)['out']
            seg_probs = torch.softmax(outputs, dim=1)
            painted_channels = seg_probs[:, [7, 15, 2], :, :] 
            
        painted_cpu = painted_channels.cpu().numpy()
        
        # Loop through the batch to do the 3D math and save
        for i in range(len(frame_nums)):
            radar_data = radars[i]
            t_camProj = t_camProjs[i]
            painted = painted_cpu[i]
            
            trans_homo_radar = np.ones((radar_data.shape[0], 4))
            trans_homo_radar[:, :3] = radar_data[:, :3]
            uv_coords_homo = (t_camProj @ trans_homo_radar.T).T
            
            u = (uv_coords_homo[:, 0] / uv_coords_homo[:, 2]).astype(int)
            v = (uv_coords_homo[:, 1] / uv_coords_homo[:, 2]).astype(int)
            
            H, W = images.shape[2], images.shape[3]
            valid_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (uv_coords_homo[:, 2] > 0)
            
            point_scores = np.zeros((radar_data.shape[0], 3), dtype=np.float32)
            point_scores[valid_mask] = painted[:, v[valid_mask], u[valid_mask]].T
            
            painted_radar = np.concatenate([radar_data, point_scores], axis=-1)
            np.save(os.path.join(save_dir, f'{frame_nums[i]}.npy'), painted_radar)
            
        # Print progress to the terminal
        if batch_idx % 5 == 0:
            print(f"Processed batch {batch_idx}/{len(dataloader)}")
            
    end_time = time.time()
    print(f"\nFinished in {(end_time - start_time) / 60:.2f} minutes.")

if __name__ == '__main__':
    preprocess_dataset(parse_args())