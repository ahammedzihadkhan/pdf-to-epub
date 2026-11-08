"""
figure_detector.py
------------------
Dual-source figure/photo detection for scanned Bengali book pages.

Strategy:
  1. PRIMARY  - Use Gemini Vision bounding boxes (from OCR JSON cache)
  2. SECONDARY - Use OpenCV morphological blob detection to verify / fill gaps
  3. Dedup overlapping regions with IoU threshold
  4. Crop, label (Bengali: chitro N), and cache each figure as a separate JPEG
     alongside a figures_index.json manifest.
"""

import sys
import json
import re
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

try:
    if sys.stdout is not None:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


def _iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    union = (ax1-ax0)*(ay1-ay0) + (bx1-bx0)*(by1-by0) - inter
    return inter / union if union > 0 else 0.0


def _deduplicate(boxes, iou_thresh=0.4):
    kept = []
    for box in boxes:
        dominated = any(_iou(box["pixel_box"], k["pixel_box"]) > iou_thresh for k in kept)
        if not dominated:
            kept.append(box)
    return kept


def _gemini_boxes(cache_file, img_w, img_h):
    if not cache_file.exists():
        return []
    try:
        data = json.loads(cache_file.read_text(encoding='utf-8'))
    except Exception:
        return []
    results = []
    for ph in data.get("photos", []):
        box = ph.get("box_2d")
        if not box:
            continue
        while len(box) == 1 and isinstance(box[0], list):
            box = box[0]
        if len(box) != 4:
            continue
        ymin, xmin, ymax, xmax = box
        x0 = max(0, int(xmin * img_w / 1000))
        y0 = max(0, int(ymin * img_h / 1000))
        x1 = min(img_w, int(xmax * img_w / 1000))
        y1 = min(img_h, int(ymax * img_h / 1000))
        if (x1 - x0) < 30 or (y1 - y0) < 30:
            continue
        results.append({
            "source": ph.get("source", "gemini"),
            "pixel_box": [x0, y0, x1, y1],
            "norm_box": box,
            "caption": ph.get("caption", ""),
        })
    return results


def detect_and_cache_figures(image_path, cache_file, figures_dir, figures_index, iou_thresh=0.35):
    """
    Detect genuine figures/photos on image_path using Gemini multimodal detections,
    crop them, save to figures_dir, and append records to figures_index list.
    Page 0000 (cover) and text paragraphs are never treated as figures.
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    image_path = Path(image_path)
    cache_file = Path(cache_file)
    page_stem = image_path.stem

    # Book cover page (page_0000) is used whole as EPUB cover, never crop into an inline figure
    if page_stem == "page_0000":
        return []

    pil_img = Image.open(image_path).convert("RGB")
    img_w, img_h = pil_img.size

    # Strictly use Gemini Vision detections to avoid false positives on normal text blocks
    all_boxes = _gemini_boxes(cache_file, img_w, img_h)
    all_boxes = _deduplicate(all_boxes, iou_thresh=iou_thresh)
    all_boxes.sort(key=lambda b: (b["pixel_box"][1], b["pixel_box"][0]))

    page_num = int(re.sub(r'\D', '', page_stem)) if any(c.isdigit() for c in page_stem) else 0
    results = []

    for fig_idx, box in enumerate(all_boxes):
        x0, y0, x1, y1 = box["pixel_box"]
        x0c = max(0, x0 - 4)
        y0c = max(0, y0 - 4)
        x1c = min(img_w, x1 + 4)
        y1c = min(img_h, y1 + 4)
        crop = pil_img.crop((x0c, y0c, x1c, y1c))

        figure_id = f"{page_stem}_fig_{fig_idx:03d}"
        fig_filename = f"{figure_id}.jpg"
        fig_path = figures_dir / fig_filename
        crop.save(str(fig_path), format="JPEG", quality=90)

        label = f"\u099a\u09bf\u09a4\u09cd\u09b0 {len(figures_index) + 1}"  # Bengali: "Chitra N"

        record = {
            "page": page_stem,
            "page_num": page_num,
            "figure_id": figure_id,
            "figure_local_idx": fig_idx,
            "source": box["source"],
            "pixel_box": box["pixel_box"],
            "norm_box": box["norm_box"],
            "file": fig_filename,
            "label": label,
            "caption": box.get("caption", ""),
        }
        figures_index.append(record)
        results.append(record)
        print(f"  [FIGURE] [{box['source'].upper()}] {figure_id}  box={box['pixel_box']}")

    return results


def save_figures_index(figures_dir, figures_index):
    figures_dir = Path(figures_dir)
    idx_path = figures_dir / "figures_index.json"
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(figures_index, f, ensure_ascii=False, indent=2)
    print(f"Figures index saved: {idx_path}  ({len(figures_index)} figures total)")


def load_figures_index(figures_dir):
    figures_dir = Path(figures_dir)
    idx_path = figures_dir / "figures_index.json"
    if idx_path.exists():
        return json.loads(idx_path.read_text(encoding="utf-8"))
    return []


if __name__ == "__main__":
    print("figure_detector.py ready.")
