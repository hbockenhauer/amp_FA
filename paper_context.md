# Project Context for Paper Writing

## Assignment Overview
This is a final assignment for the course **RO47020 Advanced Machine Perception** at TU Delft. The task is to improve a baseline radar-based 3D object detector and write a 4-page scientific paper (CVPR style). The dataset is the **View of Delft** dataset, which provides radar, camera, and LiDAR data captured from a Prius in Delft. Only radar and monocular camera are allowed as inputs (no LiDAR). The detection task is 3D bounding box prediction for three classes: Car, Pedestrian, and Cyclist. The metric is **mAP** (mean Average Precision).

---

## Baseline Model (A0)
The baseline is a **radar-adapted CenterPoint** detector. CenterPoint is a well-known LiDAR-based 3D object detector that represents objects as center points on a Bird's Eye View (BEV) heatmap. The pipeline is:

1. **Voxelization / Pillars**: The raw radar point cloud is divided into vertical columns (pillars) over the BEV plane.
2. **Pillar Feature Net (PFN)**: A small MLP encodes point-level features within each pillar into a fixed-size pillar descriptor.
3. **Sparse-to-Dense**: Pillars are scattered back onto a 2D BEV feature map.
4. **SECOND Backbone**: A 2D CNN extracts multi-scale BEV features.
5. **FPN Neck**: Multi-scale features are upsampled and concatenated.
6. **CenterPoint Heads**: Predict heatmaps for object centers plus regression of size, height, and orientation.

The radar point cloud has 7 channels: x, y, z, RCS (radar cross section), Doppler radial velocity (v_r), and 2 additional features. RCS is related to object reflectivity/size. Doppler gives direct radial velocity measurements — a unique advantage of radar over LiDAR.

**The key challenge**: radar produces only ~500–2000 points per frame vs ~100,000 for LiDAR, making the point cloud very sparse and noisy.

---

## Three Improvements

### Improvement 1: PointPainting with ResNet (A1)
**What**: Radar-camera fusion via the PointPainting technique. A pretrained **DeepLabV3-ResNet50** semantic segmentation model processes the camera image and produces per-pixel class probability scores (Car, Pedestrian, Cyclist). Each radar point is projected onto the camera image using the known sensor calibration. The class scores at the projected pixel are appended to the radar point as 3 extra channels. This increases the input from 7 to **10 channels per point**.

**Why it helps**: Radar is spatially sparse and has poor resolution — it is hard to distinguish object classes from geometry alone. The camera provides rich semantic information. By "painting" class labels onto radar points, the detector knows from the start whether a point likely belongs to a car, pedestrian, or cyclist. This addresses the fundamental limitation of radar-only detection.

**Implementation detail**: The segmentation is run as a preprocessing step (offline, before training) and the painted radar files are cached. At runtime, the painted points are loaded directly.

---

### Improvement 2: Doppler-Aware Feature Encoding (A2 + A3 → A4)
Radar's unique advantage over cameras and LiDAR is the **Doppler velocity** measurement — each point carries a direct measurement of radial velocity (toward/away from the ego vehicle). The baseline model treats Doppler as just another input channel without exploiting its structure. We introduce two complementary mechanisms:

**A2 — Doppler Cluster Embedding (in the pillar encoder)**:
Within each pillar, points are soft-assigned to a set of learnable Doppler velocity prototypes (bins). The weighted mean and variance of Doppler velocity within the pillar are computed and appended as extra point-wise features (repeated for all points in the pillar). This gives the network a compact descriptor of the *motion distribution* inside each pillar — e.g., "this pillar contains mostly fast-moving points" — which helps distinguish dynamic objects from static background clutter.

**A3 — Doppler-Guided Local Attention (in the backbone)**:
After each stage of the 2D backbone, a lightweight local self-attention module is applied to the BEV feature map. The attention between neighboring BEV cells is biased by their Doppler similarity: cells with similar Doppler values are encouraged to attend more strongly to each other. This groups BEV regions by motion coherence, helping the backbone form spatially and dynamically consistent feature representations. Optionally, the Doppler variance (reliability) is used to downweight attention contributions from uncertain cells.

**A4** combines both A2 and A3.

---

### Improvement 3: Multi-Frame Temporal Accumulation (A5)
**What**: Instead of using only the current radar frame, the model is given **5 accumulated radar frames** (the current frame plus 4 previous frames, ego-motion compensated). The View of Delft dataset provides these pre-accumulated point clouds directly.

**Why it helps**: A single radar frame may contain very few points on small or distant objects (e.g., a pedestrian may appear as 1–3 points). Accumulating multiple frames effectively densifies the point cloud, giving the model much more geometric context. The maximum number of points per pillar increases from 5 to 35, and the total voxel budget doubles.

**Trade-off**: Accumulated frames represent past positions of moving objects, not their current position. For slow-moving or static objects this is fine; for fast-moving objects there can be motion blur / smearing. This is why Doppler information (improvement 2) becomes even more valuable when combined with temporal accumulation — it helps disambiguate current from past detections.

---

## Ablation Study Design
The improvements are evaluated cumulatively:

| Config | PointPainting | Doppler Cluster | Doppler Attention | 5 Frames | Neck Refine | Wide PFN | mAP |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| A0 Baseline | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | TBD |
| A1 PointPainting | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | TBD |
| A2 +Doppler Cluster | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | TBD |
| A3 +Doppler Attention | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | TBD |
| A4 +Both Doppler | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | TBD |
| A5 +5 Frames | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | TBD |
| A6 +Neck Refine | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | TBD |
| A7 +Wide PFN | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | TBD |

Each model is trained for 12 epochs on a single A100 GPU (DelftBlue cluster, TU Delft). The evaluation metric is `Driving_corridor_area_mAP`.

---

## Scientific Story / Narrative
The central thesis is: **radar point clouds are sparse and semantically weak, but they carry unique motion information (Doppler) that is underexploited by standard LiDAR-adapted detectors**. Our three improvements each address a distinct limitation:
- PointPainting addresses **semantic weakness** by injecting camera-derived class information.
- Doppler features address **underuse of the motion channel** — the one truly radar-specific advantage.
- Temporal accumulation addresses **sparsity** by increasing effective point density.
- The wider/deeper PFN addresses the **feature extraction bottleneck** introduced by PointPainting: the original single-layer PFN is insufficient to fuse the heterogeneous radar geometric features and semantic class scores into a rich pillar representation.

Together they represent a systematic investigation of how to best adapt a LiDAR-based architecture to radar data.

---

## Paper Requirements
- 4 pages (CVPR style), excluding references
- Must include: Introduction, Related Work, Method, Experiments, Conclusion, Contributions
- At least 10 meaningful citations
- Must include one table and one figure created by the group
- Relevant prior work to cite: CenterPoint, PointPillars, PointPainting, RadarNet, nuScenes dataset paper, DeepLabV3, SECOND backbone, View of Delft dataset paper
