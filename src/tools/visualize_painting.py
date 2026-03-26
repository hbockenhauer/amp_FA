import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.models.segmentation import lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights
from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader, FrameTransformMatrix

def visualize_frame(frame_number='00000', data_root='data/view_of_delft', model_type='resnet'):
    # 1. Load Image and Radar Data
    locations = KittiLocations(root_dir=data_root)
    frame_data = FrameDataLoader(kitti_locations=locations, frame_number=frame_number)
    transforms = FrameTransformMatrix(frame_data)
    
    image = frame_data.image
    radar_data = frame_data.radar_data
    
    # 2. Run Image Segmentation
    if model_type == 'resnet':
        model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT).cuda().eval()
    elif model_type == 'mobilenet':
        model = lraspp_mobilenet_v3_large(weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT).cuda().eval()
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img_tensor = transform(image).unsqueeze(0).cuda()
    with torch.no_grad():
        output = model(img_tensor)['out'][0]
        seg_mask = torch.argmax(output, dim=0).cpu().numpy() # The predicted classes
        seg_probs = torch.softmax(output, dim=0).cpu().numpy()
        
    # 3. Project Radar Points to Camera
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
    
    # Extract the probability for "Car" (Class 7) for the valid points
    car_probabilities = seg_probs[7, valid_v, valid_u]
    
    # 4. Create the Plots
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    # Plot 1: Original Image
    axes[0].imshow(image)
    axes[0].set_title("Original Camera Image")
    axes[0].axis('off')
    
    # Plot 2: Segmentation Mask Overlay
    axes[1].imshow(image)
    axes[1].imshow(seg_mask, alpha=0.5, cmap='jet') 
    axes[1].set_title("Segmentation Mask")
    axes[1].axis('off')
    
    # Plot 3: Projected Radar Points
    axes[2].imshow(image)
    sc = axes[2].scatter(valid_u, valid_v, c=car_probabilities, cmap='coolwarm', s=15, edgecolors='black')
    axes[2].set_title(f'Painted Radar Points (Red = High Car Probability) - {model_type}')
    axes[2].axis('off')
    plt.colorbar(sc, ax=axes[2], label="Car Probability")
    
    plt.tight_layout()
    plt.savefig("pointpainting_visualization.png")
    print("Saved visualization to pointpainting_visualization.png")

if __name__ == '__main__':
    # You can change this to any valid frame number from your train.txt file
    visualize_frame('00000', model_type='resnet')