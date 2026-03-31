# Paper Ablation Study

Cumulative ablation over the improvements. A2/A3 isolate each Doppler component independently (both built on A1); A4 combines them.

| ID | Config | PointPainting (ResNet) | Doppler Cluster | Doppler Attention | Temporal (5 frames) | Neck Refine | mAP |
|----|--------|:----------------------:|:---------------:|:-----------------:|:-------------------:|:-----------:|-----|
| A0 | `centerpoint_baseline` | ✗ | ✗ | ✗ | ✗ | ✗ | |
| A1 | `pointPainting_resnet` | ✓ | ✗ | ✗ | ✗ | ✗ | |
| A2 | `pointPainting_resnet_doppler_cluster` | ✓ | ✓ | ✗ | ✗ | ✗ | |
| A3 | `pointPainting_resnet_doppler_attention` | ✓ | ✗ | ✓ | ✗ | ✗ | |
| A4 | `pointPainting_resnet_doppler` | ✓ | ✓ | ✓ | ✗ | ✗ | |
| A5 | `pointPainting_resnet_temporal_5frames_doppler` | ✓ | ✓ | ✓ | ✓ | ✗ | |
| A6 | `pointPainting_resnet_temporal_5frames_doppler_neck` | ✓ | ✓ | ✓ | ✓ | ✓ | |

Submit all runs (12 epochs each):
```bash
bash src/tools/submit_paper_ablations.sh
```
