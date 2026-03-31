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
    else: raise ValueError("painted_dir must be 'painted_radar' or 'painted_radar_resnet' or 'painted_radar_mobilenet'")

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
        # The script saves them in the order: [..., Car, Pedestrian, Cyclist]
        car_probs = painted_radar[valid_mask, -3]
        ped_probs = painted_radar[valid_mask, -2]
        cyc_probs = painted_radar[valid_mask, -1]
        
        # 5. Create the Plots
        fig, axes = plt.subplots(4, 1, figsize=(12, 20))
        
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
        
        plt.tight_layout()
        save_name = f'npy_visualization_{frame_number}_{model_type}.png'
        plt.savefig(save_name)
        plt.close(fig)
        print(f"Saved {save_name}\n")

if __name__ == '__main__':
    # Add your frames here. Change painted_dir to 'painted_radar_resnet' if needed.
    frames_to_test = ['00000', '00001', '00002', '00200', '02201', '00600', '01000'] 
    visualize_npy_files(frames_to_test, painted_dir='painted_radar_mobilenet')