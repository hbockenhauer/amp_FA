import numpy as np
import matplotlib.pyplot as plt
import os

def load_radar_points(frame_idx_str):
    """
    Placeholder function. Replace with your actual data loading code.
    """
    num_points = np.random.randint(15, 40)
    x = np.random.uniform(-40, 40, num_points)
    y = np.random.uniform(0, 50, num_points) 
    return np.column_stack((x, y))

def save_accumulated_bev(frame_indices, accumulation_steps=[1, 3, 5], save_folder="images/bev_plots"):
    """
    Generates and saves BEV plots for multiple frames and accumulation steps.
    """
    # Create a folder to save the images if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)

    for target_frame_str in frame_indices:
        target_frame_int = int(target_frame_str)
        
        for num_frames in accumulation_steps:
            plt.figure(figsize=(6, 6))
            accumulated_data = []
            
            for offset in range(num_frames):
                # Calculate past frame number, ensuring it doesn't go below 0
                past_frame_int = max(0, target_frame_int - offset)
                frame_to_load_str = f"{past_frame_int:05d}"
                
                points = load_radar_points(frame_to_load_str)
                accumulated_data.append(points)
                
            all_points = np.vstack(accumulated_data)
            
            # Create the scatter plot
            plt.scatter(all_points[:, 0], all_points[:, 1], s=15, c='darkblue', alpha=0.7)
            
            # Format the plot
            plt.xlim(-50, 50)
            plt.ylim(0, 60)
            plt.title(f"Frame {target_frame_str} | Accumulation: {num_frames}")
            plt.xlabel("X (meters)")
            plt.ylabel("Y (meters)")
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.gca().set_aspect('equal')
            plt.tight_layout()

            # Generate the specific file name
            filename = f"bev_frame_{target_frame_str}_acc_{num_frames}.png"
            filepath = os.path.join(save_folder, filename)

            # Save the image and close the plot to save memory
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved: {filename}")

# --- Execution ---
frames_to_test = ['00056', '00106', '00186', '00290', '00300', '00306', '00406', '00506', '00606', '00706', '03000', '03500', '04000', '04500', '05000', '05005', '05010', '08280', '08290', '08300', '083450', '08350', '08400', '08405']
save_accumulated_bev(frames_to_test, accumulation_steps=[1, 3, 5])