"""
Visualize predicted vs ground truth bounding boxes for all ablations on a
single validation frame, shown side by side in a BEV (top-down) plot.

Usage (run from project root on DelftBlue):
    python src/tools/visualize_ablations.py --frame_idx 42
    python src/tools/visualize_ablations.py --frame_idx 42 --data_root data/view_of_delft
    python src/tools/visualize_ablations.py --frame_idx 42 --score_thresh 0.25 --out vis_frame42.png

Arguments:
    --frame_idx    Index into the validation split (0-based). Default: 0
    --data_root    Path to the View of Delft dataset. Default: data/view_of_delft
    --score_thresh Score threshold for predicted boxes. Default: 0.2
    --out          Output PNG filename. Default: ablation_vis_frame<idx>.png
"""

import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

import torch
from omegaconf import OmegaConf

root = os.path.abspath(os.path.join(os.getcwd()))
if root not in sys.path:
    sys.path.insert(0, root)

from src.model.detector import CenterPoint
from src.dataset import ViewOfDelft, collate_vod_batch
from torch.utils.data import DataLoader

# ── Ablation registry ─────────────────────────────────────────────────────────
# Maps display name → checkpoint path. Skip any that don't exist.
ABLATIONS = [
    ('A0 Baseline',          'outputs/A0/checkpoints/last.ckpt'),
    ('A1 PointPainting',     'outputs/A1/checkpoints/last.ckpt'),
    ('A2 +Doppler Cluster',  'outputs/A2/checkpoints/last.ckpt'),
    ('A3 +Doppler Attn',     'outputs/A3/checkpoints/last.ckpt'),
    ('A4 +Both Doppler',     'outputs/A4/checkpoints/last.ckpt'),
    ('A5 +5 Frames',         'outputs/A5/checkpoints/last.ckpt'),
    ('A6 +Neck Refine',      'outputs/A6/checkpoints/last.ckpt'),
    ('A7 +Wide PFN',         'outputs/A7/checkpoints/last.ckpt'),
]

CLASS_NAMES  = ['Car', 'Pedestrian', 'Cyclist']
CLASS_COLORS = ['#FF4444', '#44FF44', '#4488FF']   # red, green, blue

# BEV view limits (metres, radar frame)
X_MIN, X_MAX = 0.0,  51.2
Y_MIN, Y_MAX = -25.6, 25.6


# ── Geometry helpers ──────────────────────────────────────────────────────────

def box_corners_bev(box):
    """Return (4, 2) array of BEV corners for a box [x, y, z, l, w, h, yaw]."""
    x, y, l, w, yaw = box[0], box[1], box[3], box[4], box[6]
    half_l, half_w = l / 2.0, w / 2.0
    # local corners (front-right, front-left, back-left, back-right)
    local = np.array([[ half_l,  half_w],
                      [ half_l, -half_w],
                      [-half_l, -half_w],
                      [-half_l,  half_w]])
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    world = local @ rot.T
    world[:, 0] += x
    world[:, 1] += y
    return world


def draw_bev_box(ax, corners, color, linewidth=1.5, linestyle='-', alpha=1.0, zorder=3):
    """Draw one oriented bounding box on a BEV axes."""
    poly = plt.Polygon(np.vstack([corners, corners[0]]),
                       closed=True, fill=False,
                       edgecolor=color, linewidth=linewidth,
                       linestyle=linestyle, alpha=alpha, zorder=zorder)
    ax.add_patch(poly)
    # heading arrow: midpoint of front edge → center
    front_mid = (corners[0] + corners[1]) / 2.0
    center    = corners.mean(axis=0)
    ax.annotate('', xy=front_mid, xytext=center,
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=linewidth, mutation_scale=8),
                zorder=zorder + 1)


# ── Dataset builder from checkpoint config ────────────────────────────────────

def build_dataset_from_ckpt(ckpt_cfg, data_root, split='val'):
    """Reconstruct the correct ViewOfDelft dataset from a checkpoint config."""
    dataset_cfg = {}
    if 'dataset' in ckpt_cfg:
        dataset_cfg = OmegaConf.to_container(ckpt_cfg['dataset'], resolve=True)

    # radar_mode lives at the model top-level for temporal configs
    if 'radar_mode' in ckpt_cfg and 'radar_mode' not in dataset_cfg:
        dataset_cfg['radar_mode'] = ckpt_cfg['radar_mode']

    return ViewOfDelft(data_root=data_root, split=split, **dataset_cfg)


# ── Single-frame inference ────────────────────────────────────────────────────

def run_inference(model, batch, score_thresh):
    """
    Run the model on one batch and return list of dicts with
    keys: boxes (N,7), scores (N,), labels (N,)  — all numpy.
    """
    model.eval()
    with torch.no_grad():
        pts_data = batch['pts']
        metas    = batch['metas']
        ret_dict, _ = model._model_forward_tta(pts_data)
        bbox_list = model.head.get_bboxes(ret_dict, img_metas=metas)

    results = []
    for bboxes, scores, labels in bbox_list:
        keep = scores >= score_thresh
        results.append(dict(
            boxes  = bboxes[keep].tensor.cpu().numpy(),   # (N, 7)
            scores = scores[keep].cpu().numpy(),
            labels = labels[keep].cpu().numpy().astype(int),
        ))
    return results


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_bev(ax, pts, pred, gt_boxes, gt_labels, title):
    """Fill one BEV subplot."""
    ax.set_facecolor('#1a1a2e')
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=8, color='white', pad=3)
    ax.tick_params(colors='#888888', labelsize=6)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')

    # Radar points — colour by Doppler if available (channel 5)
    if pts.shape[1] > 5:
        doppler = pts[:, 5]
        vmax = max(abs(doppler).max(), 0.1)
        ax.scatter(pts[:, 0], pts[:, 1],
                   c=doppler, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   s=2, alpha=0.6, linewidths=0, zorder=2)
    else:
        ax.scatter(pts[:, 0], pts[:, 1],
                   c='#aaaaaa', s=2, alpha=0.6, linewidths=0, zorder=2)

    # Ground truth boxes — dashed white
    for box, label in zip(gt_boxes, gt_labels):
        corners = box_corners_bev(box)
        draw_bev_box(ax, corners,
                     color='white', linewidth=1.2,
                     linestyle='--', alpha=0.85, zorder=4)

    # Predicted boxes — solid class colour
    if pred is not None:
        for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
            if label >= len(CLASS_COLORS):
                continue
            corners = box_corners_bev(box)
            draw_bev_box(ax, corners,
                         color=CLASS_COLORS[label], linewidth=1.5,
                         linestyle='-', alpha=0.9, zorder=5)
            # score text near box centre
            cx, cy = box[0], box[1]
            if X_MIN < cx < X_MAX and Y_MIN < cy < Y_MAX:
                ax.text(cx, cy, f'{score:.2f}',
                        fontsize=4, color=CLASS_COLORS[label],
                        ha='center', va='center', zorder=6)

    # Ego vehicle marker
    ax.plot(0, 0, 'w^', markersize=4, zorder=7)


def build_legend():
    handles = [
        mpatches.Patch(facecolor='none', edgecolor='white',
                       linestyle='--', label='Ground truth'),
    ]
    for name, color in zip(CLASS_NAMES, CLASS_COLORS):
        handles.append(mpatches.Patch(facecolor='none', edgecolor=color,
                                      label=f'Pred: {name}'))
    return handles


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_idx',   type=int,   default=0,
                        help='Index into the validation split (0-based)')
    parser.add_argument('--data_root',   type=str,   default='data/view_of_delft')
    parser.add_argument('--score_thresh',type=float, default=0.2)
    parser.add_argument('--out',         type=str,   default=None)
    args = parser.parse_args()

    output_path = args.out or f'ablation_vis_frame{args.frame_idx:04d}.png'

    # Filter to checkpoints that actually exist
    available = [(name, ckpt) for name, ckpt in ABLATIONS if os.path.isfile(ckpt)]
    if not available:
        print('No checkpoint files found. Make sure you run this from the '
              'project root and that outputs/A*/checkpoints/last.ckpt exist.')
        return

    print(f'Found {len(available)} checkpoints: {[n for n, _ in available]}')

    # ── Layout ────────────────────────────────────────────────────────────────
    n     = len(available)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 4.5),
                             facecolor='#0d0d1a')
    fig.subplots_adjust(wspace=0.05, hspace=0.15)
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    # Hide unused subplots
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    # ── Per-ablation inference ────────────────────────────────────────────────
    for plot_idx, (name, ckpt_path) in enumerate(available):
        print(f'\n[{plot_idx+1}/{n}] Loading {name} from {ckpt_path} ...')

        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        ckpt_cfg = DictConfig(ckpt['hyper_parameters']['config'])

        dataset = build_dataset_from_ckpt(ckpt_cfg, args.data_root, split='val')

        if args.frame_idx >= len(dataset):
            print(f'  frame_idx {args.frame_idx} out of range '
                  f'(dataset has {len(dataset)} val samples), skipping.')
            continue

        sample = dataset[args.frame_idx]
        batch  = collate_vod_batch([sample])

        # Move points to GPU for inference
        model = CenterPoint.load_from_checkpoint(ckpt_path, map_location='cuda')
        model.cuda().eval()

        pts_np = batch['pts'][0].numpy()  # (P, C)

        # GT boxes
        gt_boxes  = batch['gt_bboxes_3d'][0].tensor.numpy()   # (M, 7)
        gt_labels = batch['gt_labels_3d'][0].numpy()

        # Move batch pts to GPU tensors for inference
        batch_gpu = dict(
            pts         = [torch.tensor(pts_np).cuda()],
            gt_bboxes_3d= batch['gt_bboxes_3d'],
            gt_labels_3d= batch['gt_labels_3d'],
            metas       = batch['metas'],
        )

        try:
            preds = run_inference(model, batch_gpu, args.score_thresh)
            pred  = preds[0]
        except Exception as e:
            print(f'  Inference failed: {e}')
            pred = None

        frame_num = batch['metas'][0]['num_frame']
        plot_bev(axes_flat[plot_idx], pts_np, pred, gt_boxes, gt_labels,
                 title=f'{name}  |  frame {frame_num}')

        # Free GPU memory before loading next model
        del model
        torch.cuda.empty_cache()

    # ── Legend + save ─────────────────────────────────────────────────────────
    fig.legend(handles=build_legend(),
               loc='lower center', ncol=len(CLASS_NAMES) + 1,
               fontsize=8, framealpha=0.3,
               facecolor='#0d0d1a', labelcolor='white',
               bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(f'Ablation BEV Comparison — val frame idx {args.frame_idx}',
                 color='white', fontsize=12, y=1.01)

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'\nSaved to {output_path}')


if __name__ == '__main__':
    from omegaconf import DictConfig
    main()
