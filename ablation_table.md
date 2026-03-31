# Paper Ablation Study

Cumulative ablation over the three improvements. Each run adds exactly one component on top of the previous.

| ID | Config | PointPainting (ResNet) | Doppler | Temporal (5 frames) | mAP |
|----|--------|:----------------------:|:-------:|:-------------------:|-----|
| A0 | `centerpoint_baseline` | ✗ | ✗ | ✗ | |
| A1 | `pointPainting_resnet` | ✓ | ✗ | ✗ | |
| A2 | `pointPainting_resnet_doppler` | ✓ | ✓ | ✗ | |
| A3 | `pointPainting_resnet_temporal_5frames_doppler` | ✓ | ✓ | ✓ | |

Submit all runs (20 epochs each):
```bash
bash src/tools/submit_paper_ablations.sh
```
