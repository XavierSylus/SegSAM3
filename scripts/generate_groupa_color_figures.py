#!/usr/bin/env python3
"""
Generate qualitative color figures for GroupA archive.

Fixes:
1) Strictly validates --archive_dir must be a directory.
2) Robust binary-mask loading for RGB/RGBA images (ignores alpha-only pitfalls).
3) Outputs per-sample metrics and best/worst picks for thesis use.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PRED_TOKENS = ("pred", "prediction", "output")
GT_TOKENS = ("gt", "groundtruth", "label", "target")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GroupA qualitative color figures.")
    parser.add_argument("--archive_dir", type=str, required=True, help="e.g. thesis_artifacts/group_a_YYYYMMDD_HHMMSS")
    parser.add_argument("--num_samples", type=int, default=10, help="number of qualitative figures to export")
    parser.add_argument("--seed", type=int, default=42, help="random seed for deterministic tie-break")
    return parser.parse_args()


def validate_archive_dir(path_str: str) -> Path:
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"--archive_dir does not exist: {p}")
    if not p.is_dir():
        raise NotADirectoryError(f"--archive_dir must be a directory, got file: {p}")
    suffix_joined = "".join(p.suffixes)
    if suffix_joined in {".tar.gz", ".tar.gz.sha256", ".sha256"}:
        raise NotADirectoryError(f"--archive_dir points to archive/hash file, not directory: {p}")
    return p


def normalize_key(name: str) -> str:
    stem = Path(name).stem.lower()
    stem = re.sub(r"[_\-\s]+", "_", stem)
    stem = re.sub(r"(?:^|_)(pred|prediction|output|gt|groundtruth|label|target|mask)(?:_|$)", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem


def classify_file(path: Path) -> str:
    n = path.stem.lower()
    has_pred = any(tok in n for tok in PRED_TOKENS)
    has_gt = any(tok in n for tok in GT_TOKENS)
    if has_pred and not has_gt:
        return "pred"
    if has_gt and not has_pred:
        return "gt"
    return "unknown"


def collect_pairs(mask_root: Path) -> List[Tuple[str, Path, Path]]:
    files = [p for p in mask_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    gt_map: Dict[str, Path] = {}
    pred_map: Dict[str, Path] = {}

    for p in files:
        role = classify_file(p)
        if role == "unknown":
            continue
        key = normalize_key(p.name)
        if role == "gt":
            gt_map[key] = p
        elif role == "pred":
            pred_map[key] = p

    keys = sorted(set(gt_map).intersection(pred_map))
    return [(k, gt_map[k], pred_map[k]) for k in keys]


def load_binary_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 2:
        gray = arr
    elif arr.ndim == 3:
        rgb = arr[:, :, :3]
        rgb_gray = rgb.max(axis=2)
        if arr.shape[2] >= 4:
            alpha = arr[:, :, 3]
            # Avoid "all foreground" when alpha channel is constant 255.
            if np.ptp(rgb_gray) == 0 and np.ptp(alpha) > 0:
                gray = alpha
            else:
                gray = rgb_gray
        else:
            gray = rgb_gray
    else:
        raise ValueError(f"Unsupported image shape for mask: {path} -> {arr.shape}")

    # Conservative threshold for binary mask.
    return (gray > 127).astype(np.uint8)


def dice_iou(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    inter = np.logical_and(pred_b, gt_b).sum()
    pred_sum = pred_b.sum()
    gt_sum = gt_b.sum()
    union = pred_sum + gt_sum - inter

    eps = 1e-8
    dice = (2.0 * inter + eps) / (pred_sum + gt_sum + eps)
    iou = (inter + eps) / (union + eps)
    return float(dice), float(iou)


def render_triptych(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    h, w = gt.shape
    tp = (pred == 1) & (gt == 1)
    fp = (pred == 1) & (gt == 0)
    fn = (pred == 0) & (gt == 1)

    gt_vis = np.zeros((h, w, 3), dtype=np.uint8)
    gt_vis[gt == 1] = np.array([255, 255, 255], dtype=np.uint8)

    pred_vis = np.zeros((h, w, 3), dtype=np.uint8)
    pred_vis[pred == 1] = np.array([255, 255, 0], dtype=np.uint8)

    diff_vis = np.zeros((h, w, 3), dtype=np.uint8)
    diff_vis[tp] = np.array([0, 255, 0], dtype=np.uint8)      # TP: green
    diff_vis[fp] = np.array([255, 0, 0], dtype=np.uint8)      # FP: red
    diff_vis[fn] = np.array([0, 0, 255], dtype=np.uint8)      # FN: blue

    sep = np.full((h, 8, 3), 20, dtype=np.uint8)
    return np.concatenate([gt_vis, sep, pred_vis, sep, diff_vis], axis=1)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    archive_dir = validate_archive_dir(args.archive_dir)
    mask_root = archive_dir / "segmentation_masks"
    if not mask_root.exists() or not mask_root.is_dir():
        raise FileNotFoundError(f"Missing segmentation_masks directory: {mask_root}")

    out_dir = archive_dir / "figures" / "qualitative_color"
    metrics_dir = archive_dir / "metrics"
    paper_dir = archive_dir / "paper"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs(mask_root)
    if not pairs:
        raise RuntimeError(
            "No GT/PRED pairs found in segmentation_masks. "
            "Expected names containing tokens like 'gt' and 'pred'."
        )

    rows = []
    for key, gt_path, pred_path in pairs:
        gt = load_binary_mask(gt_path)
        pred = load_binary_mask(pred_path)
        if gt.shape != pred.shape:
            # Strict resize-free policy for metrics consistency.
            continue

        gt_pixels = int(gt.sum())
        pred_pixels = int(pred.sum())
        ratio = gt_pixels / float(gt.size)

        # Skip clearly corrupted/all-foreground GT masks.
        if ratio > 0.95:
            continue

        d, i = dice_iou(pred, gt)
        rows.append(
            {
                "index": key,
                "dice": d,
                "iou": i,
                "pred_pixels": pred_pixels,
                "gt_pixels": gt_pixels,
                "gt_path": str(gt_path),
                "pred_path": str(pred_path),
            }
        )

    if not rows:
        raise RuntimeError("No valid samples after mask sanity checks.")

    rows.sort(key=lambda x: x["dice"], reverse=True)
    n = min(args.num_samples, len(rows))
    half = n // 2
    selected = rows[:half] + rows[-(n - half):]

    # Export figures.
    for r in selected:
        gt = load_binary_mask(Path(r["gt_path"]))
        pred = load_binary_mask(Path(r["pred_path"]))
        canvas = render_triptych(gt, pred)
        Image.fromarray(canvas).save(out_dir / f"{r['index']}_triptych.png")

    # Export CSV.
    csv_path = metrics_dir / "qual_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "dice", "iou", "pred_pixels", "gt_pixels", "gt_path", "pred_path"])
        for r in rows:
            writer.writerow([r["index"], f"{r['dice']:.8f}", f"{r['iou']:.8f}", r["pred_pixels"], r["gt_pixels"], r["gt_path"], r["pred_path"]])

    # Export selection markdown.
    best3 = rows[:3]
    worst3 = rows[-3:]

    md_sel = paper_dir / "qualitative_selection.md"
    md_sel.write_text(
        "\n".join(
            [
                "# GroupA 定性样例候选",
                "",
                f"- 有效样本数: {len(rows)}",
                f"- 导出图像数: {len(selected)}",
                "- 图例: GT=白色, Pred=黄色, TP=绿色, FP=红色, FN=蓝色",
                "",
                "## Best 3",
                *[
                    f"- `{r['index']}` Dice={r['dice']:.4f}, IoU={r['iou']:.4f}, Pred={r['pred_pixels']}, GT={r['gt_pixels']}"
                    for r in best3
                ],
                "",
                "## Worst 3",
                *[
                    f"- `{r['index']}` Dice={r['dice']:.4f}, IoU={r['iou']:.4f}, Pred={r['pred_pixels']}, GT={r['gt_pixels']}"
                    for r in worst3
                ],
            ]
        ),
        encoding="utf-8",
    )

    md_final = paper_dir / "qualitative_final_pick.md"
    md_final.write_text(
        "\n".join(
            [
                "# GroupA 论文最终定性样本（建议）",
                "",
                "## 建议入正文（3 优 + 3 劣）",
                *[
                    f"- `{r['index']}` Dice={r['dice']:.4f}, 图像: `figures/qualitative_color/{r['index']}_triptych.png`"
                    for r in (best3 + worst3)
                ],
            ]
        ),
        encoding="utf-8",
    )

    print(f"[OK] saved figures -> {out_dir}")
    print(f"[OK] saved metrics -> {csv_path}")
    print(f"[OK] saved selection -> {md_sel}")
    print(f"[OK] saved final pick -> {md_final}")


if __name__ == "__main__":
    main()
