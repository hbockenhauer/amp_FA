import os
import sys
import numpy as np
from vod.frame import FrameDataLoader, FrameTransformMatrix
from vod.configuration import KittiLocations

# Import your dataset class to access the live painting function
print(os.getcwd())
sys.path.append(os.getcwd())
from src.dataset import ViewOfDelft, collate_vod_batch

def check_painting_consistency(frame_number='00000', data_root='data/view_of_delft'):
    print(f"Testing frame {frame_number}...")

    # 1. Load Preprocessed Data (.npy)
    npy_path = os.path.join(os.getcwd(), 'painted_radar_resnet', f"{frame_number}.npy")
    if not os.path.exists(npy_path):
        print(f"Error: Could not find {npy_path}")
        return
        
    radar_preprocessed = np.load(npy_path)

    # 2. Run Live Processing
    # Initialize the dataset class just to use its painting function
    dataset = ViewOfDelft(
        data_root=data_root, 
        use_painted_radar=True, 
        painted_radar_dir='painted_radar_resnet'
    )
    
    locations = KittiLocations(root_dir=data_root)
    frame_data = FrameDataLoader(kitti_locations=locations, frame_number=frame_number)
    transforms = FrameTransformMatrix(frame_data)
    raw_radar = frame_data.radar_data

    # Call the fallback method manually
    radar_live = dataset._paint_radar_inference(frame_data, transforms, raw_radar)

    # 3. Compare Results
    print(f"Preprocessed shape: {radar_preprocessed.shape}")
    print(f"Preprocessed content: {radar_preprocessed}")
    print(f"Live shape:         {radar_live.shape}")
    print(f"Live content:         {radar_live}")

    # Use allclose to compare floats (allows for tiny 0.00001 rounding differences)
    is_same = np.allclose(radar_preprocessed, radar_live, atol=1e-5)

    if is_same:
        print("Success! The live processing matches the preprocessed file exactly.")
    else:
        print("Difference found! The arrays do not match.")
        max_diff = np.max(np.abs(radar_preprocessed - radar_live))
        print(f"Maximum value difference: {max_diff}")

if __name__ == '__main__':
    # You can change the frame number to test different files
    check_painting_consistency('05000')