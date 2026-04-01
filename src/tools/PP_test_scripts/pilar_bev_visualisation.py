# srun --partition=gpu-a100-small --account=education-me-courses-ro47020 --time=03:00:00 --ntasks=1 --cpus-per-task=2 --mem-per-cpu=4G --gpus-per-task=1 python src/tools/PP_test_scripts/pilar_bev_visualisation.py checkpoint_path=outputs/PP_ResNet_doppler_3F/checkpoints/last.ckpt model=pointPainting_resnet_temporal_3frames_doppler
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Ensure src module is in the path
root = os.path.abspath(os.path.join(os.getcwd()))
if root not in sys.path:
    sys.path.insert(0, root)

import hydra
from omegaconf import DictConfig
from src.model.detector import CenterPoint
from src.dataset import ViewOfDelft, collate_vod_batch

@hydra.main(version_base=None, config_path='../../config', config_name="test")
def visualize_pillars(cfg: DictConfig) -> None:
    # 1. Setup Model and Dataset
    dataset = ViewOfDelft(data_root=cfg.data_root, split='val', radar_mode=cfg.model.radar_mode)
    model = CenterPoint.load_from_checkpoint(checkpoint_path=cfg.checkpoint_path)
    model.eval().cuda()

    # Get Voxel config
    voxel_size = cfg.model.voxel_size # e.g., [0.32, 0.32, 5]
    pc_range = cfg.model.point_cloud_range # e.g., [0, -25.6, -3, 51.2, 25.6, 2]

    # 2. Load a single frame (e.g., frame '00006')
    frame_idx = dataset.sample_list.index('05000')
    batch = collate_vod_batch([dataset[frame_idx]])
    pts = [p.cuda() for p in batch['pts']]

    # 3. Get Voxel Data from the model
    with torch.no_grad():
        voxel_dict = model.voxelize(pts)
    
    # coors shape: [Num_Voxels, 4] -> (batch_idx, z_idx, y_idx, x_idx)
    coors = voxel_dict['coors'].cpu().numpy()
    voxels = voxel_dict['voxels'].cpu().numpy()
    num_points = voxel_dict['num_points'].cpu().numpy()

    # 4. Create the Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot active pillars (squares)
    for i in range(len(coors)):
        # Calculate real-world coordinates from grid indices
        y_idx, x_idx = coors[i, 2], coors[i, 3]
        
        # Real-world X and Y = (Index * Voxel_Size) + Point_Cloud_Min_Range
        x_real = (x_idx * voxel_size[0]) + pc_range[0]
        y_real = (y_idx * voxel_size[1]) + pc_range[1]
        
        # Draw the pillar box
        rect = patches.Rectangle(
            (x_real, y_real), voxel_size[0], voxel_size[1], 
            linewidth=1, edgecolor='blue', facecolor='lightblue', alpha=0.4
        )
        ax.add_patch(rect)

        # Plot the actual points inside this specific pillar
        # voxels[i] contains the points for this pillar. 
        # Feature 0 is X, Feature 1 is Y.
        valid_points = num_points[i]
        pillar_pts_x = voxels[i, :valid_points, 0]
        pillar_pts_y = voxels[i, :valid_points, 1]
        
        ax.scatter(pillar_pts_x, pillar_pts_y, s=5, c='red', zorder=5)

    # 5. Format Plot
    ax.set_title("BEV Point Pillars Visualization")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_xlim(pc_range[0], pc_range[3])
    ax.set_ylim(pc_range[1], pc_range[4])
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_aspect('equal')

    # Save
    save_path = "pillar_visualization.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved pillar plot to {save_path}")

if __name__ == '__main__':
    visualize_pillars()