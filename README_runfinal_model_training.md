# Submit Final Model

Runs preprocessing and training back-to-back in a single GPU allocation (4 h wall time).

## How it works

1. Generates `painted_radar_resnet_5frames/` using the ResNet PointPainting pipeline (skipped if it already exists)
2. Trains the final model for 24 epochs using that data

## Usage

```bash
sbatch src/tools/submit_a7.sh
```
