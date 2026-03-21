import argparse
import os

import numpy as np
from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader


def summarize_columns(arr):
    print("Array shape:", arr.shape)
    if arr.ndim != 2:
        print("Expected a 2D radar array [num_points, num_features].")
        return

    print("\nPer-column stats:")
    print("idx\tmin\tmax\tmean\tstd\tfrac_pos\tfrac_neg")
    candidates = []
    for i in range(arr.shape[1]):
        col = arr[:, i]
        finite = np.isfinite(col)
        if not finite.any():
            print(f"{i}\tNaN\tNaN\tNaN\tNaN\tNaN\tNaN")
            continue

        c = col[finite]
        cmin = float(np.min(c))
        cmax = float(np.max(c))
        cmean = float(np.mean(c))
        cstd = float(np.std(c))
        frac_pos = float(np.mean(c > 0))
        frac_neg = float(np.mean(c < 0))

        print(
            f"{i}\t{cmin:.3f}\t{cmax:.3f}\t{cmean:.3f}\t{cstd:.3f}\t{frac_pos:.3f}\t{frac_neg:.3f}"
        )

        # Heuristic: Doppler should have both signs and realistic speed range.
        has_both_signs = frac_pos > 0.05 and frac_neg > 0.05
        plausible_range = -80.0 < cmin < 0.0 and 0.0 < cmax < 80.0
        not_too_large_std = cstd < 30.0
        if has_both_signs and plausible_range and not_too_large_std:
            candidates.append(i)

    if candidates:
        print("\nPossible Doppler columns (heuristic):", candidates)
    else:
        print("\nNo clear Doppler candidate found with default heuristic.")


def main():
    parser = argparse.ArgumentParser(description="Inspect View-of-Delft radar feature columns")
    parser.add_argument("--data-root", default="data/view_of_delft", help="Path to View-of-Delft root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Split file")
    parser.add_argument("--index", type=int, default=0, help="Frame index in split file")
    args = parser.parse_args()

    split_file = os.path.join(args.data_root, "lidar", "ImageSets", f"{args.split}.txt")
    with open(split_file, "r", encoding="utf-8") as f:
        frames = [line.strip() for line in f if line.strip()]

    if args.index < 0 or args.index >= len(frames):
        raise IndexError(f"index {args.index} out of range [0, {len(frames)-1}]")

    frame = frames[args.index]
    kitti_locations = KittiLocations(root_dir=args.data_root)
    loader = FrameDataLoader(kitti_locations=kitti_locations, frame_number=frame)

    radar_data = np.asarray(loader.radar_data)
    print("Frame:", frame)
    summarize_columns(radar_data)

    if radar_data.ndim == 2 and radar_data.shape[0] > 0:
        print("\nFirst radar point:")
        print(radar_data[0])


if __name__ == "__main__":
    main()
