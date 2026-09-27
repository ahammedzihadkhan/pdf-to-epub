import sys
import os
import json
import re
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

ROOT_DIR = Path(__file__).parent.parent
PAGES_DIR = ROOT_DIR / "pages"
CACHE_DIR = ROOT_DIR / "ocr_cache"
DATASET_DIR = Path(__file__).parent / "dataset"
IMAGES_DIR = DATASET_DIR / "images"

def extract_text_lines_from_markdown(md_text: str) -> list:
    """Clean markdown text into individual natural sentences/lines for OCR training."""
    clean = re.sub(r'#+\s*', '', md_text)
    clean = re.sub(r'\*\*(.*?)\*\*', r'\1', clean)
    clean = re.sub(r'\*(.*?)\*', r'\1', clean)
    clean = re.sub(r'\[\^.*?\]', '', clean)
    clean = re.sub(r'>\s*', '', clean)
    
    paragraphs = [p.strip() for p in clean.split('\n') if p.strip()]
    lines = []
    for p in paragraphs:
        sentences = [s.strip() for s in re.split(r'(?<=[।?!])\s+', p) if len(s.strip()) > 5]
        if sentences:
            lines.extend(sentences)
        else:
            lines.append(p)
    return lines

def segment_image_lines(image_path: Path) -> list:
    """Segment printed document page into horizontal text line crops using morphological analysis."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []
    
    h, w = img.shape
    top_margin = int(h * 0.08)
    bottom_margin = int(h * 0.92)
    left_margin = int(w * 0.06)
    right_margin = int(w * 0.94)
    
    cropped = img[top_margin:bottom_margin, left_margin:right_margin]
    ch, cw = cropped.shape
    
    _, thresh = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cw // 15, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[1])
    
    line_crops = []
    for x, y, bw, bh in bounding_boxes:
        if bh >= 18 and bw >= cw // 6:
            pad_y = 4
            pad_x = 8
            y1 = max(0, y - pad_y)
            y2 = min(ch, y + bh + pad_y)
            x1 = max(0, x - pad_x)
            x2 = min(cw, x + bw + pad_x)
            
            line_img = cropped[y1:y2, x1:x2]
            line_crops.append(line_img)
            
    return line_crops

def build_ocr_dataset(max_pages=200):
    """Build paired image-text OCR training dataset from 616 book pages."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    page_files = sorted(PAGES_DIR.glob("page_*.jpg"))
    if max_pages:
        page_files = page_files[:max_pages]
        
    print(f"📦 Extracting OCR Training Pairs from {len(page_files)} pages...")
    
    records = []
    total_lines = 0
    
    for idx, p_path in enumerate(page_files):
        stem = p_path.stem
        json_path = CACHE_DIR / f"{stem}.json"
        
        if not json_path.exists():
            continue
            
        try:
            data = json.loads(json_path.read_text(encoding='utf-8'))
            md = data.get("content_markdown", "") or ""
        except Exception:
            continue
            
        text_lines = extract_text_lines_from_markdown(md)
        image_crops = segment_image_lines(p_path)
        
        n_pairs = min(len(text_lines), len(image_crops))
        
        for i in range(n_pairs):
            txt = text_lines[i]
            img_crop = image_crops[i]
            
            if len(txt) < 8 or len(txt) > 200:
                continue
                
            img_filename = f"line_{total_lines:06d}.jpg"
            img_save_path = IMAGES_DIR / img_filename
            
            cv2.imwrite(str(img_save_path), img_crop)
            
            records.append({
                "file_name": f"images/{img_filename}",
                "text": txt,
                "page": stem
            })
            total_lines += 1
            
        if (idx + 1) % 20 == 0 or idx == len(page_files) - 1:
            print(f"  Processed {idx + 1}/{len(page_files)} pages ➔ {total_lines} training line pairs generated.")

    np.random.seed(42)
    np.random.shuffle(records)
    
    split_idx = int(len(records) * 0.85)
    train_records = records[:split_idx]
    val_records = records[split_idx:]
    
    with open(DATASET_DIR / "train.jsonl", "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    with open(DATASET_DIR / "val.jsonl", "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"\n🎉 Dataset Generation Complete!")
    print(f"   📁 Output Directory: {DATASET_DIR}")
    print(f"   📊 Total Samples: {len(records):,}")
    print(f"   🏋️ Train Samples: {len(train_records):,} (train.jsonl)")
    print(f"   🧪 Validation Samples: {len(val_records):,} (val.jsonl)")

if __name__ == "__main__":
    build_ocr_dataset(max_pages=200)
