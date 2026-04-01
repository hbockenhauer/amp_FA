# srun --partition=gpu-a100-small --account=education-me-courses-ro47020 --time=03:00:00 --ntasks=1 --cpus-per-task=2 --mem-per-cpu=4G --gpus-per-task=1 python src/tools/PP_test_scripts/visualize_painting.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.models.segmentation import lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights

import sys
root = os.path.abspath(os.path.join(os.getcwd()))
if root not in sys.path:
    sys.path.insert(0, root)

from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader, FrameTransformMatrix

def visualize_multiple_frames(frame_numbers, data_root='data/view_of_delft', model_type='resnet', radar_mode='single'):
    # Define radar folders based on your ViewOfDelft dataset class
    RADAR_MODES = {
        'single':   'radar',
        '3_frames': 'radar_3frames',
        '5_frames': 'radar_5frames',
    }
    
    if radar_mode not in RADAR_MODES:
        raise ValueError(f"radar_mode must be one of {list(RADAR_MODES.keys())}")

    # 1. Load the Model Once
    if model_type == 'resnet':
        model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT).cuda().eval()
    elif model_type == 'mobilenet':
        model = lraspp_mobilenet_v3_large(weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT).cuda().eval()
    else:
        raise ValueError("model_type must be 'resnet' or 'mobilenet'")
        
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    locations = KittiLocations(root_dir=data_root)
    # Redirect radar directory to the correct temporal folder
    radar_folder = RADAR_MODES[radar_mode]
    locations.radar_dir = os.path.join(data_root, radar_folder, 'training', 'velodyne')

    # 2. Loop Through All Frames
    for frame_number in frame_numbers:
        print(f"Processing frame {frame_number} with {model_type} ({radar_mode})...")
        
        # Load Data
        frame_data = FrameDataLoader(kitti_locations=locations, frame_number=frame_number)
        transforms = FrameTransformMatrix(frame_data)
        
        image = frame_data.image
        radar_data = frame_data.radar_data
        
        # Run Image Segmentation
        img_tensor = transform(image).unsqueeze(0).cuda()
        with torch.no_grad():
            output = model(img_tensor)['out'][0]
            seg_mask = torch.argmax(output, dim=0).cpu().numpy() # The predicted classes
            seg_probs = torch.softmax(output, dim=0).cpu().numpy()
            
        # Project Radar Points to Camera
        trans_homo_radar = np.ones((radar_data.shape[0], 4))
        trans_homo_radar[:, :3] = radar_data[:, :3]
        
        t_camProj_lidar = transforms.camera_projection_matrix @ transforms.t_camera_lidar
        uv_coords_homo = (t_camProj_lidar @ trans_homo_radar.T).T
        
        u = (uv_coords_homo[:, 0] / uv_coords_homo[:, 2]).astype(int)
        v = (uv_coords_homo[:, 1] / uv_coords_homo[:, 2]).astype(int)
        
        H, W = image.shape[:2]
        valid_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (uv_coords_homo[:, 2] > 0)
        
        valid_u = u[valid_mask]
        valid_v = v[valid_mask]
        
        # Extract probabilities for the valid points
        car_probs = seg_probs[7, valid_v, valid_u]       # Class 7: Car
        ped_probs = seg_probs[15, valid_v, valid_u]      # Class 15: Person
        cyc_probs = seg_probs[2, valid_v, valid_u]       # Class 2: Bicycle
        
        # 3. Create the Plots
        fig, axes = plt.subplots(5, 1, figsize=(12, 25))
        
        # Plot 0: Original Image
        axes[0].imshow(image)
        axes[0].set_title(f"Original Camera Image (Frame {frame_number})")
        axes[0].axis('off')
        
        # Plot 1: Segmentation Mask Overlay
        axes[1].imshow(image)
        axes[1].imshow(seg_mask, alpha=0.5, cmap='jet') 
        axes[1].set_title(f"Segmentation Mask - {model_type}")
        axes[1].axis('off')
        
        # Plot 2: Car Probabilities (Red)
        axes[2].imshow(image)
        sc1 = axes[2].scatter(valid_u, valid_v, c=car_probs, cmap='Reds', s=20, edgecolors='black')
        axes[2].set_title(f'Car Probability - {model_type} (Radar: {radar_mode})')
        axes[2].axis('off')
        plt.colorbar(sc1, ax=axes[2])
        
        # Plot 3: Pedestrian Probabilities (Green)
        axes[3].imshow(image)
        sc2 = axes[3].scatter(valid_u, valid_v, c=ped_probs, cmap='Greens', s=20, edgecolors='black')
        axes[3].set_title(f'Pedestrian Probability - {model_type} (Radar: {radar_mode})')
        axes[3].axis('off')
        plt.colorbar(sc2, ax=axes[3])

        # Plot 4: Cyclist Probabilities (Blue)
        axes[4].imshow(image)
        sc3 = axes[4].scatter(valid_u, valid_v, c=cyc_probs, cmap='Blues', s=20, edgecolors='black')
        axes[4].set_title(f'Cyclist Probability - {model_type} (Radar: {radar_mode})')
        axes[4].axis('off')
        plt.colorbar(sc3, ax=axes[4])
        
        plt.tight_layout()
        save_name = f'pointpainting_vis_{model_type}_{radar_mode}_{frame_number}.png'
        plt.savefig(save_name)
        plt.close(fig) 
        print(f"Saved visualization to {save_name}\n")

if __name__ == '__main__':
    frames_to_test = ['00106', '00290', '00300', '00306', '03000', '03500', '04000', '04500', '05000', '05005', '05010', '08300', '08350', '08400'] # Example frame numbers
    
    # You can change model_type to 'mobilenet' and radar_mode to 'single', '3_frames', or '5_frames'
    visualize_multiple_frames(
        frames_to_test, 
        model_type='mobilenet', 
        radar_mode='single'
    )