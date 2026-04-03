# Paper Ablation Study

Cumulative ablation over the improvements. A2/A3 isolate each Doppler component independently (both built on A1); A4 combines them.

ROI = Driving corridor area. All scores reported at the epoch with best entire-area mAP (model checkpoint criterion).

| ID | Config | PointPainting (ResNet) | Doppler Cluster | Doppler Attention | Temporal (5 frames) | Neck Refine | Wide PFN | Car (full) | Ped (full) | Cyc (full) | mAP (full) | Car (ROI) | Ped (ROI) | Cyc (ROI) | mAP (ROI) |
|----|--------|:----------------------:|:---------------:|:-----------------:|:-------------------:|:-----------:|:--------:|-----------:|-----------:|-----------:|-----------:|----------:|----------:|----------:|----------:|
| A0 | `centerpoint_baseline` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 19.82 | 17.55 | 48.81 | 28.73 | 54.83 | 24.38 | 72.28 | 50.50 |
| A1 | `pointPainting_resnet` | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | 24.07 | 19.36 | 51.48 | 31.64 | 58.31 | 25.42 | 74.67 | 52.80 |
| A2 | `pointPainting_resnet_doppler_cluster` | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | 24.40 | 20.33 | 56.09 | 33.61 | 58.54 | 26.95 | 79.17 | 54.89 |
| A3 | `pointPainting_resnet_doppler_attention` | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | 24.20 | 19.40 | 50.47 | 31.36 | 59.05 | 24.36 | 73.08 | 52.16 |
| A4 | `pointPainting_resnet_doppler` | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | 24.79 | 19.83 | 49.30 | 31.31 | 66.04 | 27.98 | 71.02 | 55.01 |
| A5 | `pointPainting_resnet_temporal_5frames_doppler` | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | 30.45 | 23.57 | 57.18 | 37.07 | 66.12 | 25.66 | 84.51 | 58.76 |
| A6 | `pointPainting_resnet_temporal_5frames_doppler_neck` | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | 29.92 | 23.72 | 51.69 | 35.11 | 69.13 | 28.67 | 68.58 | 55.46 |
| A7 | `pointPainting_resnet_temporal_5frames_doppler_neck_wide_pfn` | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | 27.83 | 24.06 | 58.28 | 36.72 | 61.20 | 31.33 | 83.68 | 58.73 |
| A7wide | `pointPainting_resnet_temporal_5frames_doppler_neck_wide_pfn` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 28.13 | 24.22 | 58.35 | 36.90 | 66.95 | 32.32 | 82.56 | 60.61 |
| A8 | `pointPainting_resnet_temporal_5frames_doppler_wide_pfn` | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | | | | | | | | |

Submit all runs (12 epochs each):
```bash
bash src/tools/submit_paper_ablations.sh
```
