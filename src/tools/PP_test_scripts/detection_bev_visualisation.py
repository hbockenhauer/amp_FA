# USAGE example
# 
# srun --partition=gpu-a100-small --account=education-me-courses-ro47020 --time=03:00:00 --ntasks=1 --cpus-per-task=2 --mem-per-cpu=4G --gpus-per-task=1 python src/tools/PP_test_scripts/detection_bev_visualisation.py checkpoint_path=outputs/PP_ResNet_doppler_3F/checkpoints/last.ckpt model=pointPainting_resnet_temporal_3frames_doppler
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
from omegaconf import DictConfig, OmegaConf
from src.model.detector import CenterPoint
from src.dataset import ViewOfDelft, collate_vod_batch
from vod.frame import FrameDataLoader

def draw_bev_box(ax, box, color, linestyle='-', label=None):
    """Helper function to draw a 2D bounding box in BEV from 3D box center and dimensions."""
    x, y = box[0], box[1]
    dx, dy = box[3], box[4]
    yaw = box[6]

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    # Calculate 4 corners relative to the center
    x_corners = np.array([dx/2, dx/2, -dx/2, -dx/2])
    y_corners = np.array([dy/2, -dy/2, -dy/2, dy/2])

    # Rotate and translate to actual coordinates
    x_rot = x + x_corners * cos_yaw - y_corners * sin_yaw
    y_rot = y + x_corners * sin_yaw + y_corners * cos_yaw

    pts = np.stack([x_rot, y_rot], axis=1)
    poly = patches.Polygon(pts, closed=True, linewidth=2, edgecolor=color, facecolor='none', linestyle=linestyle)
    ax.add_patch(poly)
    
    if label:
        ax.text(x_rot[0], y_rot[0] + 0.5, label, color=color, fontsize=9, weight='bold')

@hydra.main(version_base=None, config_path='../../config', config_name="test")
def generate_qualitative_plots(cfg: DictConfig) -> None:
    print('Initializing qualitative results generation...')
    
    # 1. Dataset Setup
    dataset_cfg = {}
    if 'dataset' in cfg.model:
        dataset_cfg = OmegaConf.to_container(cfg.model.dataset, resolve=True)
        print(dataset_cfg)
    if 'radar_mode' in cfg.model and 'radar_mode' not in dataset_cfg:
        dataset_cfg['radar_mode'] = cfg.model.radar_mode

    # Use the validation split so we have access to Ground Truth labels
    dataset = ViewOfDelft(
        data_root=cfg.data_root,
        split='val',
        **dataset_cfg)

    # 2. Model Setup
    print(f"Loading model from checkpoint: {cfg.checkpoint_path}")
    model = CenterPoint.load_from_checkpoint(checkpoint_path=cfg.checkpoint_path)
    model.eval()
    model.cuda()
    model.inference_mode = 'val' 

    # Extract model name from checkpoint path for file naming
    model_name = cfg.model.get('name', 'centerpoint')
    print("model name:", model_name)

    save_folder = os.path.join(os.path.dirname(os.path.dirname(cfg.checkpoint_path)), "qualitative_plots")
    print("save folder:", save_folder)
    os.makedirs(save_folder, exist_ok=True)

    # 3. Choose frames to visualize, 08500
    frames_to_test = ['00306', '05000', '08400']
    frame_to_idx = {frame: i for i, frame in enumerate(dataset.sample_list)}

    for frame in frames_to_test:
        if frame not in frame_to_idx:
            print(f"Frame {frame} not found in val split. Skipping.")
            continue

        # Load data for the specific frame
        idx = frame_to_idx[frame]
        data_dict = dataset[idx]
        batch = collate_vod_batch([data_dict])

        # Move to GPU
        batch['pts'] = [p.cuda() for p in batch['pts']]
        batch['gt_labels_3d'] = [l.cuda() for l in batch['gt_labels_3d']]
        batch['gt_bboxes_3d'] = [b.to('cuda') for b in batch['gt_bboxes_3d']]

        # 4. Run Inference
        with torch.no_grad():
            ret_dict, _ = model._model_forward_tta(batch['pts'])
            bbox_list = model.head.get_bboxes(ret_dict, img_metas=batch['metas'])

        bbox_results = [
            dict(bboxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
            for bboxes, scores, labels in bbox_list
        ]

        # Use CenterPoint's built-in converter to format valid boxes
        pred_dict = model.convert_valid_bboxes(bbox_results[0], batch)

        # 5. Extract Ground Truth Information
        vod_frame_data = FrameDataLoader(kitti_locations=dataset.vod_kitti_locations, frame_number=frame)
        image = vod_frame_data.image
        
        # Parse Ground Truth 2D directly from raw labels
        gt_2d_boxes = []
        for label_str in vod_frame_data.raw_labels:
            parts = label_str.split(' ')
            if parts[0] in dataset.CLASSES:
                left, top, right, bottom = map(float, parts[4:8])
                gt_2d_boxes.append([left, top, right - left, bottom - top])

        # Ground Truth 3D tensor from dataset output
        gt_3d_tensor = batch['gt_bboxes_3d'][0].tensor.cpu().numpy()
        points = batch['pts'][0].cpu().numpy()
        
        # 6. Create Plotting Layout
        fig, (ax_img, ax_bev) = plt.subplots(1, 2, figsize=(16, 7))

        # --- Camera Image Plot ---
        ax_img.imshow(image)
        ax_img.set_title(f"Camera View (2D Bounding Boxes) - Frame {frame}")
        ax_img.axis('off')

        # Draw GT 2D Boxes (Green, slightly transparent)
        for box in gt_2d_boxes:
            x, y, w, h = box
            # Added alpha=0.6 for transparency
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='green', facecolor='none', alpha=0.6)
            ax_img.add_patch(rect)

        # Draw Predicted 2D Boxes (Red)
        if len(pred_dict['box2d']) > 0:
            for box, score, lbl_idx in zip(pred_dict['box2d'], pred_dict['scores'], pred_dict['label_preds']):
                if score < 0.3: continue 
                min_x, min_y, max_x, max_y = box
                w, h = max_x - min_x, max_y - min_y
                cls_name = model.class_names[int(lbl_idx)]
                
                # Added alpha=0.6 for transparency
                rect = patches.Rectangle((min_x, min_y), w, h, linewidth=2, edgecolor='red', facecolor='none', linestyle='--', alpha=0.8)
                ax_img.add_patch(rect)
                
                # Removed 'backgroundcolor' so the text has no white border
                ax_img.text(min_x, min_y - 5, f"{cls_name} {score:.2f}", color='red', fontsize=10, weight='bold')

        # --- BEV Map Plot ---
        ax_bev.scatter(points[:, 0], points[:, 1], s=2, c='darkgray', alpha=0.8)
        ax_bev.set_title(f"Bird's Eye View (3D Boxes) - {model_name}")
        ax_bev.set_xlim(0, 50) # Point cloud typically looks forward
        ax_bev.set_ylim(-25, 25)
        ax_bev.set_xlabel("X (meters)")
        ax_bev.set_ylabel("Y (meters)")
        ax_bev.grid(True, linestyle='--', alpha=0.6)
        ax_bev.set_aspect('equal')

        # Draw GT 3D Boxes (Green)
        for box in gt_3d_tensor:
            draw_bev_box(ax_bev, box, color='green', linestyle='-')

        # Draw Predicted 3D Boxes (Red)
        if len(pred_dict['box3d_lidar']) > 0:
            for box, score, lbl_idx in zip(pred_dict['box3d_lidar'], pred_dict['scores'], pred_dict['label_preds']):
                if score < 0.3: continue
                cls_name = model.class_names[int(lbl_idx)]
                draw_bev_box(ax_bev, box, color='red', linestyle='--', label=f"{cls_name} {score:.2f}")

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', lw=2, label='Ground Truth'),
            Line2D([0], [0], color='red', lw=2, linestyle='--', label='Prediction')
        ]
        ax_bev.legend(handles=legend_elements, loc='upper right')

        # 7. Save Output Image
        plt.tight_layout()
        filename = f"qualitative_{frame}_{model_name}.png"
        filepath = os.path.join(save_folder, filename)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filepath}")

if __name__ == '__main__':
    generate_qualitative_plots()