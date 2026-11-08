#!/usr/bin/env python3
"""
export_training_dataset.py
==========================
Converts the Gemini OCR cache into a structured training dataset for future
Bangla OCR / Vision-LLM fine-tuning and evaluation.

Output layout
--------------
<out_dir>/
    dataset_card.md          <- HuggingFace-style dataset card
    metadata.json            <- Book/source metadata
    train.jsonl              <- 90% training split
    val.jsonl                <- 10% validation split
    all.jsonl                <- Full dataset (all pages)
"""

import json
import shutil
import random
import argparse
from pathlib import Path
from datetime import datetime


DATASET_CARD_TEMPLATE = """---
language:
- bn
license: cc-by-4.0
task_categories:
- image-to-text
tags:
- bangla
- ocr
- bengali
---

# Bangla OCR Training Dataset: {source_book}

Extracted from '{source_book}' by {author} using Gemini Vision OCR pipeline.

## Fields
- id, source_book, author, language, ocr_model
- page_index, image_path
- heading, raw_text, content_markdown
- figures (list of {{id, path, box, detector, label}})
- has_figures

## Splits
| Split | Count | Ratio |
|-------|-------|-------|
| train | {train_count} | 90%   |
| val   | {val_count} | 10%   |
| all   | {total_count} | 100%  |
"""


def load_figures_index(figures_dir: Path):
    idx_file = figures_dir / "figures_index.json"
    if not idx_file.exists():
        return {}
    raw = json.loads(idx_file.read_text(encoding="utf-8"))
    by_page = {}
    for fig in raw:
        key = fig.get("page", "")
        by_page.setdefault(key, []).append(fig)
    return by_page


def build_records(cache_dir: Path, pages_dir: Path, figures_by_page: dict, book_meta: dict):
    cache_files = sorted(cache_dir.glob("page_*.json"))
    records = []
    source_prefix = book_meta["source_book"]

    for cf in cache_files:
        page_stem = cf.stem
        try:
            page_idx = int(page_stem.split("_")[1])
        except Exception:
            page_idx = 0

        try:
            ocr = json.loads(cf.read_text(encoding="utf-8"))
        except Exception:
            continue

        img_candidates = list(pages_dir.glob(f"{page_stem}.*"))
        img_rel = f"images/{img_candidates[0].name}" if img_candidates else None
        fig_records = figures_by_page.get(page_stem, [])
        fig_export  = []
        for i, fig in enumerate(fig_records):
            fig_export.append({
                "id":       fig.get("figure_id", fig.get("id", f"{page_stem}_fig_{i:03d}")),
                "path":     f"figures/{fig.get('file', fig.get('id',''))}",
                "box":      fig.get("norm_box", fig.get("pixel_box", [])),
                "detector": fig.get("source", "gemini"),
                "label":    fig.get("label", f"Figure {i+1}"),
                "caption":  fig.get("caption", ""),
            })

        record = {
            "id":               f"{source_prefix}_{page_stem}",
            **book_meta,
            "page_index":       page_idx,
            "image_path":       img_rel,
            "heading":          ocr.get("heading", ""),
            "raw_text":         ocr.get("raw_text", ""),
            "content_markdown": ocr.get("content_markdown", ""),
            "figures":          fig_export,
            "has_figures":      len(fig_export) > 0,
        }
        records.append(record)
    return records


def write_splits(records, out_dir: Path, val_ratio=0.10):
    random.seed(42)
    shuffled = records.copy()
    random.shuffle(shuffled)
    split_idx = max(1, int(len(shuffled) * (1 - val_ratio)))
    splits = {"train": shuffled[:split_idx], "val": shuffled[split_idx:], "all": records}
    for name, data in splits.items():
        path = out_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for rec in data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  OK  {name}.jsonl  ({len(data)} records)")
    return len(splits["train"]), len(splits["val"]), len(records)


def main():
    parser = argparse.ArgumentParser(description="Export structured OCR training dataset")
    parser.add_argument("--cache-dir",    default="mahatirtha_ocr_cache", help="Path to OCR JSON cache")
    parser.add_argument("--pages-dir",    default="mahatirtha_pages",     help="Path to page image directory")
    parser.add_argument("--figures-dir",  default="mahatirtha_figures",   help="Path to figures directory")
    parser.add_argument("--out-dir",      default="mahatirtha_dataset",   help="Output directory for training dataset")
    parser.add_argument("--source-book",  default="মহাতীর্থের শেষ যাত্রী",   help="Book title identifier")
    parser.add_argument("--author",       default="বিমল দে",               help="Author name")
    parser.add_argument("--val-ratio",    type=float, default=0.10,       help="Validation split ratio")
    args = parser.parse_args()

    cache_dir   = Path(args.cache_dir)
    pages_dir   = Path(args.pages_dir)
    figures_dir = Path(args.figures_dir)
    out_dir     = Path(args.out_dir)

    print(f"\n=== Exporting Training Dataset for: {args.source_book} ===")
    if not cache_dir.exists() or not any(cache_dir.glob("page_*.json")):
        print(f"ERROR: No OCR cache found at {cache_dir}.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    book_meta = {
        "source_book": args.source_book,
        "author":      args.author,
        "language":    "bn",
        "ocr_model":   "gemini-3.5-flash",
        "exported_at": datetime.now().isoformat(),
    }

    print("[1/3] Loading OCR cache and figure records...")
    figures_by_page = load_figures_index(figures_dir)
    records = build_records(cache_dir, pages_dir, figures_by_page, book_meta)
    print(f"  Loaded {len(records)} page records ({sum(r['has_figures'] for r in records)} pages with figures)")

    print("[2/3] Writing JSONL splits (train.jsonl, val.jsonl, all.jsonl)...")
    train_cnt, val_cnt, total_cnt = write_splits(records, out_dir, val_ratio=args.val_ratio)

    print("[3/3] Writing metadata.json & dataset_card.md...")
    pages_with_figs = sum(1 for r in records if r["has_figures"])
    meta = {
        **book_meta,
        "total_pages": total_cnt,
        "pages_with_figures": pages_with_figs,
        "total_figures": sum(len(r["figures"]) for r in records),
        "train_size": train_cnt,
        "val_size": val_cnt,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    
    card_content = DATASET_CARD_TEMPLATE.format(
        source_book=args.source_book,
        author=args.author,
        train_count=train_cnt,
        val_count=val_cnt,
        total_count=total_cnt,
    )
    (out_dir / "dataset_card.md").write_text(card_content, encoding="utf-8")

    print(f"\n=== Training Dataset Saved Successfully! ===")
    print(f"  Location    : {out_dir.resolve()}")
    print(f"  Total Pages : {meta['total_pages']}")
    print(f"  Train / Val : {meta['train_size']} / {meta['val_size']}")
    print(f"  Metadata    : {out_dir / 'metadata.json'}")
    print(f"  Splits      : {out_dir / 'train.jsonl'}, {out_dir / 'val.jsonl'}, {out_dir / 'all.jsonl'}")


if __name__ == "__main__":
    main()
