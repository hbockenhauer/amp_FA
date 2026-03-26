import os
import sys
import time
import numpy as np

# Add project root to path
root = os.path.abspath(os.path.join(os.getcwd()))
if root not in sys.path:
    sys.path.insert(0, root)

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights

from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader, FrameTransformMatrix

# 1. Custom Dataset to load files using background workers
class VoDPreprocessDataset(Dataset):
    def __init__(self, data_root, sample_list):
        self.sample_list = sample_list
        self.vod_kitti_locations = KittiLocations(root_dir=data_root)
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

def preprocess_dataset():
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 3. Load the fast MobileNet model
    seg_model = lraspp_mobilenet_v3_large(weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT).to(device).eval()
    
    data_root = 'data/view_of_delft'
    save_dir = os.path.join(os.getcwd(), 'painted_radar')
    os.makedirs(save_dir, exist_ok=True)
    
    all_frames = []
    for split in ['train', 'val', 'test']:
        split_file = os.path.join(data_root, 'lidar', 'ImageSets', f'{split}.txt')
        with open(split_file, 'r') as f:
            all_frames.extend([line.strip() for line in f.readlines()])
            
    # Skip files you already processed in the slow run
    frames_to_process = [f for f in all_frames if not os.path.exists(os.path.join(save_dir, f'{f}.npy'))]
    print(f"Frames left to process: {len(frames_to_process)}")
    
    dataset = VoDPreprocessDataset(data_root, frames_to_process)
    
    # Multiprocessing and Batching
    dataloader = DataLoader(dataset, batch_size=6, num_workers=2, collate_fn=custom_collate, prefetch_factor=1)
    
    for batch_idx, (images, radars, t_camProjs, frame_nums) in enumerate(dataloader):
        images = images.to(device)
        
        # Process 16 images at once
        with torch.no_grad():
            outputs = seg_model(images)['out']
            seg_probs = torch.softmax(outputs, dim=1)
            painted_channels = seg_probs[:, [3, 1, 2], :, :] 
            
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
    preprocess_dataset()