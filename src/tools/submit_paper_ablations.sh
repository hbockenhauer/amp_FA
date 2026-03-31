#!/bin/bash
set -euo pipefail

# Submit all paper ablation jobs to DelftBlue.
# For ablations that need painted radar (A1–A6), a preprocessing job is
# submitted first and training depends on it finishing successfully.
#
# Ablation chain:
#  A0 - Baseline           : CenterPoint radar, single frame
#  A1 - +PointPainting     : ResNet PointPainting
#  A2 - +Doppler Cluster   : pillar-level Doppler clustering only
#  A3 - +Doppler Attention : backbone Doppler attention only
#  A4 - +Doppler Both      : clustering + attention
#  A5 - +Temporal          : 5-frame radar accumulation
#  A6 - +Neck              : neck refinement module (= full model)
#
# Usage:
#   bash src/tools/submit_paper_ablations.sh
#   EPOCHS=12 BATCH_SIZE=4 bash src/tools/submit_paper_ablations.sh
#   RUN_TAG=run2 bash src/tools/submit_paper_ablations.sh

DATA_ROOT="${DATA_ROOT:-data/view_of_delft}"

# ── Preprocessing dependencies ────────────────────────────────────────────────
# painted_radar_resnet (single frame) — needed by A1, A2, A3, A4
PP_SINGLE_JID=""
if [ ! -d "painted_radar_resnet" ]; then
  echo "painted_radar_resnet not found — submitting preprocessing job (single frame)"
  PP_SINGLE_JID=$(sbatch --parsable \
    --export=ALL,PREPROCESS_MODEL=resnet,PREPROCESS_RADAR_MODE=single \
    src/tools/slurm_preprocess_pointpainting.sh)
  echo "  Preprocessing job (single): $PP_SINGLE_JID"
else
  echo "painted_radar_resnet already exists — skipping preprocessing for A1–A4"
fi

# painted_radar_resnet_5frames — needed by A5, A6
PP_5F_JID=""
if [ ! -d "painted_radar_resnet_5frames" ]; then
  echo "painted_radar_resnet_5frames not found — submitting preprocessing job (5 frames)"
  PP_5F_JID=$(sbatch --parsable \
    --export=ALL,PREPROCESS_MODEL=resnet,PREPROCESS_RADAR_MODE=5_frames \
    src/tools/slurm_preprocess_pointpainting.sh)
  echo "  Preprocessing job (5frames): $PP_5F_JID"
else
  echo "painted_radar_resnet_5frames already exists — skipping preprocessing for A5–A6"
fi

# ── Helper to build --dependency flag ────────────────────────────────────────
dep_flag() {
  local jid="$1"
  if [ -n "$jid" ]; then
    echo "--dependency=afterok:$jid"
  fi
}

# ── Submit training jobs ──────────────────────────────────────────────────────
echo ""
echo "Submitting A0 (no preprocessing required)"
sbatch --export=ALL,ABLATION_ID=A0 src/tools/slurm_paper_ablation.sh
sleep 1

for ID in A1 A2 A3 A4; do
  echo "Submitting $ID $([ -n "$PP_SINGLE_JID" ] && echo "(after preprocess job $PP_SINGLE_JID)")"
  sbatch $(dep_flag "$PP_SINGLE_JID") \
    --export=ALL,ABLATION_ID="$ID" src/tools/slurm_paper_ablation.sh
  sleep 1
done

for ID in A5 A6; do
  echo "Submitting $ID $([ -n "$PP_5F_JID" ] && echo "(after preprocess job $PP_5F_JID)")"
  sbatch $(dep_flag "$PP_5F_JID") \
    --export=ALL,ABLATION_ID="$ID" src/tools/slurm_paper_ablation.sh
  sleep 1
done

echo ""
echo "Submitted 7 training jobs (A0–A6)"
echo "Monitor with: squeue -u \$USER"
