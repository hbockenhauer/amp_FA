import argparse
import csv
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class AblationResult:
    ablation_id: str
    exp_id: str
    mode: str
    file: str
    entire_map: Optional[float]
    roi_map: Optional[float]
    car_3d: Optional[float]
    pedestrian_3d: Optional[float]
    cyclist_3d: Optional[float]


def _find_first(pattern: str, text: str, cast=float):
    m = re.search(pattern, text, flags=re.MULTILINE)
    if not m:
        return None
    return cast(m.group(1))


def _find_last(pattern: str, text: str, cast=float):
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if not matches:
        return None
    last = matches[-1]
    # If pattern has multiple capture groups, re.findall returns tuples.
    if isinstance(last, tuple):
        last = last[0]
    return cast(last)


def parse_log(path: str) -> Optional[AblationResult]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    exp_id = _find_first(r"as\s+([A-Za-z0-9_\-]+)", text, cast=str)
    ablation_id = _find_first(r"ablation\s+([A-Za-z0-9_\-]+)\s+as", text, cast=str)
    mode = "eval" if "Running eval" in text else "train"

    if exp_id is None:
        exp_id = _find_first(r"exp_id=([A-Za-z0-9_\-]+)", text, cast=str)
    if ablation_id is None and exp_id is not None:
        ablation_id = exp_id

    entire_map = None
    roi_map = None
    car_3d = None
    pedestrian_3d = None
    cyclist_3d = None

    # Parse explicit result prints from on_validation_epoch_end.
    entire_map = _find_last(r"Entire annotated area:\s*[\s\S]*?mAP:\s*([0-9]+(?:\.[0-9]+)?)", text)
    roi_map = _find_last(r"Driving corridor area:\s*[\s\S]*?mAP:\s*([0-9]+(?:\.[0-9]+)?)", text)

    car_3d = _find_last(r"Driving corridor area:\s*[\s\S]*?Car:\s*([0-9]+(?:\.[0-9]+)?)", text)
    pedestrian_3d = _find_last(r"Driving corridor area:\s*[\s\S]*?Pedestrian:\s*([0-9]+(?:\.[0-9]+)?)", text)
    cyclist_3d = _find_last(r"Driving corridor area:\s*[\s\S]*?Cyclist:\s*([0-9]+(?:\.[0-9]+)?)", text)

    # Fallbacks for lightning-style scalar printouts.
    if roi_map is None:
        roi_map = _find_last(r"validation/ROI/mAP[^0-9]*([0-9]+(?:\.[0-9]+)?)", text)
    if entire_map is None:
        entire_map = _find_last(r"validation/entire_area/mAP[^0-9]*([0-9]+(?:\.[0-9]+)?)", text)

    # If this file has no identifiable experiment or metrics, skip it.
    if exp_id is None and roi_map is None and entire_map is None:
        return None

    return AblationResult(
        ablation_id=ablation_id or "unknown",
        exp_id=exp_id or "unknown",
        mode=mode,
        file=os.path.basename(path),
        entire_map=entire_map,
        roi_map=roi_map,
        car_3d=car_3d,
        pedestrian_3d=pedestrian_3d,
        cyclist_3d=cyclist_3d,
    )


def dedupe_best(results: List[AblationResult]) -> List[AblationResult]:
    grouped: Dict[str, AblationResult] = {}
    for r in results:
        key = r.ablation_id
        score = r.roi_map if r.roi_map is not None else -1.0
        if key not in grouped:
            grouped[key] = r
        else:
            prev = grouped[key]
            prev_score = prev.roi_map if prev.roi_map is not None else -1.0
            if score > prev_score:
                grouped[key] = r
    return list(grouped.values())


def write_csv(path: str, results: List[AblationResult]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ablation_id",
            "exp_id",
            "mode",
            "roi_map",
            "entire_map",
            "car_3d",
            "pedestrian_3d",
            "cyclist_3d",
            "source_log",
        ])
        for r in results:
            writer.writerow([
                r.ablation_id,
                r.exp_id,
                r.mode,
                "" if r.roi_map is None else f"{r.roi_map:.6f}",
                "" if r.entire_map is None else f"{r.entire_map:.6f}",
                "" if r.car_3d is None else f"{r.car_3d:.6f}",
                "" if r.pedestrian_3d is None else f"{r.pedestrian_3d:.6f}",
                "" if r.cyclist_3d is None else f"{r.cyclist_3d:.6f}",
                r.file,
            ])


def write_markdown(path: str, results: List[AblationResult]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = []
    lines.append("# Ablation Results Summary")
    lines.append("")
    lines.append("| Ablation | Experiment | ROI mAP | Entire mAP | Car | Pedestrian | Cyclist | Log |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")

    for r in results:
        roi = "-" if r.roi_map is None else f"{r.roi_map:.4f}"
        entire = "-" if r.entire_map is None else f"{r.entire_map:.4f}"
        car = "-" if r.car_3d is None else f"{r.car_3d:.4f}"
        ped = "-" if r.pedestrian_3d is None else f"{r.pedestrian_3d:.4f}"
        cyc = "-" if r.cyclist_3d is None else f"{r.cyclist_3d:.4f}"
        lines.append(
            f"| {r.ablation_id} | {r.exp_id} | {roi} | {entire} | {car} | {ped} | {cyc} | {r.file} |"
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Collect ablation metrics from DelftBlue SLURM logs")
    parser.add_argument("--logs-dir", default="outputs/slurm_logs", help="Directory with .out/.err files")
    parser.add_argument("--csv", default="outputs/ablation_summary.csv", help="Output CSV path")
    parser.add_argument("--md", default="outputs/ablation_summary.md", help="Output Markdown path")
    parser.add_argument("--keep-all", action="store_true", help="Do not deduplicate per ablation id")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.logs_dir, "*.out")))
    if not files:
        files = sorted(glob.glob(os.path.join(args.logs_dir, "*.err")))

    results: List[AblationResult] = []
    for p in files:
        res = parse_log(p)
        if res is not None:
            results.append(res)

    if not results:
        print("No ablation results could be parsed.")
        return

    if not args.keep_all:
        results = dedupe_best(results)

    results.sort(key=lambda r: (-(r.roi_map if r.roi_map is not None else -1.0), r.ablation_id))

    write_csv(args.csv, results)
    write_markdown(args.md, results)

    print(f"Wrote CSV: {args.csv}")
    print(f"Wrote Markdown: {args.md}")


if __name__ == "__main__":
    main()
