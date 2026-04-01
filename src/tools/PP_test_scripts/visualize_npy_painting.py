# python src/tools/PP_test_scripts/visualize_npy_painting.py

import os
import numpy as np
import matplotlib.pyplot as plt
from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader, FrameTransformMatrix

def visualize_npy_files(frame_numbers, data_root='data/view_of_delft', painted_dir='painted_radar'):
    locations = KittiLocations(root_dir=data_root)

    if painted_dir == 'painted_radar': model_type = 'mobilenet'
    elif painted_dir == 'painted_radar_resnet': model_type = 'resnet'
    elif painted_dir == 'painted_radar_mobilenet': model_type = 'mobilenet'
    elif 'resnet_3frames' in painted_dir: model_type = 'resnet_3frames'
    elif 'resnet_5frames' in painted_dir: model_type = 'resnet_5frames'
    else: raise ValueError(f"Unknown painted_dir: {painted_dir}")

    # Determine plot settings based on the model type
    is_temporal = 'frames' in model_type
    num_plots = 5 if is_temporal else 4
    fig_height = 25 if is_temporal else 20
    
    if model_type == 'resnet_3frames':
        min_time = -2
    elif model_type == 'resnet_5frames':
        min_time = -4
    else:
        min_time = 0

    for frame_number in frame_numbers:
        print(f"Loading preprocessed frame {frame_number}...")
        
        # 1. Load Camera Image
        frame_data = FrameDataLoader(kitti_locations=locations, frame_number=frame_number)
        transforms = FrameTransformMatrix(frame_data)
        image = frame_data.image
        
        # 2. Load Preprocessed .npy File
        npy_path = os.path.join(painted_dir, f"{frame_number}.npy")
        if not os.path.exists(npy_path):
            print(f"Could not find {npy_path}. Skipping.")
            continue
            
        painted_radar = np.load(npy_path)
        
        # 3. Project Points to Camera
        trans_homo_radar = np.ones((painted_radar.shape[0], 4))
        trans_homo_radar[:, :3] = painted_radar[:, :3]
        
        t_camProj_lidar = transforms.camera_projection_matrix @ transforms.t_camera_lidar
        uv_coords_homo = (t_camProj_lidar @ trans_homo_radar.T).T
        
        u = (uv_coords_homo[:, 0] / uv_coords_homo[:, 2]).astype(int)
        v = (uv_coords_homo[:, 1] / uv_coords_homo[:, 2]).astype(int)
        
        H, W = image.shape[:2]
        valid_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (uv_coords_homo[:, 2] > 0)
        
        valid_u = u[valid_mask]
        valid_v = v[valid_mask]
        
        # 4. Extract Precalculated Probabilities
        car_probs = painted_radar[valid_mask, -3]
        ped_probs = painted_radar[valid_mask, -2]
        cyc_probs = painted_radar[valid_mask, -1]
        
        # 5. Create the Plots
        fig, axes = plt.subplots(num_plots, 1, figsize=(12, fig_height))
        
        axes[0].imshow(image)
        axes[0].set_title(f"Original Camera Image (Frame {frame_number}, Model: {model_type})")
        axes[0].axis('off')
        
        axes[1].imshow(image)
        sc1 = axes[1].scatter(valid_u, valid_v, c=car_probs, cmap='Reds', s=20, edgecolors='black')
        axes[1].set_title("Training Data: Car Probability")
        axes[1].axis('off')
        plt.colorbar(sc1, ax=axes[1])
        
        axes[2].imshow(image)
        sc2 = axes[2].scatter(valid_u, valid_v, c=ped_probs, cmap='Greens', s=20, edgecolors='black')
        axes[2].set_title("Training Data: Pedestrian Probability")
        axes[2].axis('off')
        plt.colorbar(sc2, ax=axes[2])

        axes[3].imshow(image)
        sc3 = axes[3].scatter(valid_u, valid_v, c=cyc_probs, cmap='Blues', s=20, edgecolors='black')
        axes[3].set_title("Training Data: Cyclist Probability")
        axes[3].axis('off')
        plt.colorbar(sc3, ax=axes[3])
        
        # Only create the 5th plot if it is a multi-frame dataset
        if is_temporal:
            time_index = painted_radar[valid_mask, 6].astype(int)
            axes[4].imshow(image)
            
            sc4 = axes[4].scatter(valid_u, valid_v, c=time_index, cmap='Set1', s=20, edgecolors='black', vmin=min_time, vmax=0)
            axes[4].set_title(f"Temporal Data: Points Colored by Frame Age (0 to {min_time})")
            axes[4].axis('off')
            
            # Dynamically create the correct ticks (e.g., [0, -1, -2] or [0, -1, -2, -3, -4])
            ticks = list(range(0, min_time - 1, -1))
            cbar = plt.colorbar(sc4, ax=axes[4], ticks=ticks)
            cbar.set_label(f'Frame Age (0 to {min_time})')
        
        plt.tight_layout()
        save_name = f'npy_visualization_{frame_number}_{model_type}.png'
        plt.savefig(save_name)
        plt.close(fig)
        print(f"Saved {save_name}\n")

if __name__ == '__main__':
    # Add your frames here. Change painted_dir to 'painted_radar_resnet' if needed.
    #frames_to_test = ['00000', '00001', '00002', '00200', '02201', '00600', '01000']
    frames_to_test = ['00106', '00290', '00300', '00306', '03000', '03500', '04000', '04500', '05000', '05005', '05010', '08300', '08350', '08400']
    visualize_npy_files(frames_to_test, painted_dir='painted_radar_resnet')
    #_3frames