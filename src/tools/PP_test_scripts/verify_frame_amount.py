# verify_frame_bug.py
import os
import sys
root = os.path.abspath(os.path.join(os.getcwd()))
if root not in sys.path:
    sys.path.insert(0, root)

print(root)
from omegaconf import OmegaConf
from src.dataset.view_of_delft import ViewOfDelft

def check_true_dimensions():
    # 1. Test Single Frame Painted
    cfg_single = {
        'radar_mode': 'single',
        'use_painted_radar': True,
        'painted_radar_dir': 'painted_radar_resnet'
    } 
    dataset_single = ViewOfDelft(data_root="data/view_of_delft", split="val", **cfg_single)
    
    # 2. Test 3-Frame Painted
    cfg_3frames = {
        'radar_mode': '3_frames',
        'use_painted_radar': True,
        'painted_radar_dir': 'painted_radar_resnet_3frames'
    } 
    dataset_3frames = ViewOfDelft(data_root="data/view_of_delft", split="val", **cfg_3frames)

    # 3. Test 5-Frame Painted
    cfg_5frames = {
        'radar_mode': '5_frames',
        'use_painted_radar': True,
        'painted_radar_dir': 'painted_radar_resnet_5frames'
    } 
    dataset_5frames = ViewOfDelft(data_root="data/view_of_delft", split="val", **cfg_5frames)
    
    # Pick index 10 so it has previous frames to accumulate
    test_idx = 10 
    
    print(f"--- Testing Validation Index {test_idx} ---")
    
    data_single = dataset_single[test_idx]['lidar_data']
    print(f"Single Frame mode: {dataset_single.radar_mode}")
    print(f"Single Frame Shape: {data_single.shape} (Expected: 10 dims)")
    
    data_3frames = dataset_3frames[test_idx]['lidar_data']
    print(f"3-Frame mode: {dataset_3frames.radar_mode}")
    print(f"3-Frame Shape: {data_3frames.shape} (Expected: ~3x points, 10 dims)")

    data_5frames = dataset_5frames[test_idx]['lidar_data']
    print(f"5-Frame mode: {dataset_5frames.radar_mode}")
    print(f"5-Frame Shape: {data_5frames.shape} (Expected: ~3x points, 10 dims)")

def check_test_set():
    split_file = "data/view_of_delft/lidar/ImageSets/test.txt"
    if not os.path.exists(split_file):
        print("Test split file not found.")
        return

    with open(split_file, "r") as f:
        test_frames = [line.strip() for line in f.readlines()]

    folders = [
        "painted_radar_resnet", 
        "painted_radar_resnet_3frames", 
        "painted_radar_resnet_5frames",
        "painted_radar_mobilenet"
    ]

    for folder in folders:
        if not os.path.exists(folder):
            print(f"Folder {folder} does not exist yet.")
            continue
            
        missing = sum(1 for frame in test_frames if not os.path.exists(os.path.join(folder, f"{frame}.npy")))
        print(f"Folder {folder}: {missing} missing files out of {len(test_frames)} test frames.")

if __name__ == '__main__':
    check_true_dimensions()
    check_test_set()
