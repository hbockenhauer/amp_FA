import os
import sys
# Ensure src module is in the path
root = os.path.abspath(os.path.join(os.getcwd()))
if root not in sys.path:
    sys.path.insert(0, root)
import matplotlib.pyplot as plt
import numpy as np
import torch
from src.dataset.view_of_delft import ViewOfDelft

def draw_box_bev(ax, box, color='g'):
    """Draws a 2D BEV bounding box with a red heading indicator."""
    x, y, z, dx, dy, dz, yaw = box[:7]
    
    # Define corners (assuming dx=length, dy=width)
    corners = np.array([
        [ dx/2,  dy/2],
        [-dx/2,  dy/2],
        [-dx/2, -dy/2],
        [ dx/2, -dy/2],
        [ dx/2,  dy/2] # close the loop
    ])
    
    # Rotate corners
    rot_mat = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw),  np.cos(yaw)]
    ])
    rotated_corners = np.dot(corners, rot_mat.T)
    
    # Translate to box center
    rotated_corners[:, 0] += x
    rotated_corners[:, 1] += y
    
    # Plot the box outline
    ax.plot(rotated_corners[:, 0], rotated_corners[:, 1], color=color, linewidth=2)
    
    # Draw heading line (red line pointing front)
    front = np.array([[dx/2, 0]])
    rotated_front = np.dot(front, rot_mat.T)
    ax.plot([x, x + rotated_front[0, 0]], [y, y + rotated_front[0, 1]], color='r', linewidth=2)

def main():
    print("Loading dataset...")
    dataset = ViewOfDelft(data_root='data/view_of_delft', split='train', radar_mode='5_frames')
    
    # Create output directory
    os.makedirs("batch_bev_plots", exist_ok=True)
    
    # Number of frames to check
    num_frames_to_check = 10 
    
    for idx in range(num_frames_to_check):
        data_dict = dataset[idx] 
        frame_id = data_dict['meta']['num_frame']
        
        radar_points = data_dict['lidar_data'].numpy()
        gt_boxes = data_dict['gt_bboxes_3d'].tensor.numpy()
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title(f"BEV GT Check - Frame: {frame_id}")
        ax.set_xlabel("X (forward) [m]")
        ax.set_ylabel("Y (left) [m]")
        
        # Plot radar points (grey dots)
        ax.scatter(radar_points[:, 0], radar_points[:, 1], c='grey', s=5, label='Radar Points')
        
        # Plot GT boxes
        has_objects = False
        for box in gt_boxes:
            if np.sum(box) != 0:
                draw_box_bev(ax, box)
                has_objects = True
                
        ax.set_xlim(0, 50)
        ax.set_ylim(-25, 25)
        ax.grid(True)
        
        output_path = f"batch_bev_plots/frame_{frame_id}.png"
        plt.savefig(output_path, dpi=300)
        plt.close(fig)
        
        print(f"[{idx+1}/{num_frames_to_check}] Saved {output_path} (Objects present: {has_objects})")

if __name__ == "__main__":
    main()