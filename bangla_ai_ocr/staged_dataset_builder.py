import os
import sys
import json
import random
from pathlib import Path
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

DATASET_DIR = Path(__file__).parent / "dataset"
IMAGES_DIR = DATASET_DIR / "images"

def build_staged_datasets():
    print("==================================================")
    print("📦 BUILDING STAGED OCR DATASETS & HOLDOUT TEST SETS")
    print("==================================================")
    
    train_file = DATASET_DIR / "train.jsonl"
    val_file = DATASET_DIR / "val.jsonl"
    
    records = []
    for f in [train_file, val_file]:
        if f.exists():
            with open(f, "r", encoding="utf-8") as inf:
                for line in inf:
                    if line.strip():
                        records.append(json.loads(line))
                        
    print(f"Total available samples: {len(records)}")
    
    # Partition Printed vs Handwritten
    printed_samples = [r for r in records if "dummy" not in r.get("file_name", "")]
    handwritten_samples = [r for r in records if "handwritten" in r.get("file_name", "")]
    
    if not handwritten_samples:
        handwritten_samples = printed_samples[-300:]
        printed_samples = printed_samples[:-300]
        
    random.seed(42)
    random.shuffle(printed_samples)
    random.shuffle(handwritten_samples)
    
    # 1. Carve out a 15% Pristine Holdout Test Set (NEVER seen during training)
    holdout_size_p = int(len(printed_samples) * 0.15)
    holdout_size_h = int(len(handwritten_samples) * 0.15)
    
    test_pristine = printed_samples[:holdout_size_p] + handwritten_samples[:holdout_size_h]
    rem_printed = printed_samples[holdout_size_p:]
    rem_handwritten = handwritten_samples[holdout_size_h:]
    
    # 2. Split remainder into Train and Validation
    val_size_p = int(len(rem_printed) * 0.12)
    val_size_h = int(len(rem_handwritten) * 0.12)
    
    train_printed = rem_printed[val_size_p:]
    val_printed = rem_printed[:val_size_p]
    
    train_handwritten = rem_handwritten[val_size_h:]
    val_handwritten = rem_handwritten[:val_size_h]
    
    val_unified = val_printed + val_handwritten
    
    # 3. Create Noisy Holdout Test Set metadata
    test_noisy = []
    for item in test_pristine:
        test_noisy.append({
            "file_name": item["file_name"],
            "text": item["text"],
            "is_corrupted": True
        })
        
    # Write files
    (DATASET_DIR / "train_printed.jsonl").write_text(
        "\n".join([json.dumps(r, ensure_ascii=False) for r in train_printed]), encoding="utf-8"
    )
    (DATASET_DIR / "train_handwritten.jsonl").write_text(
        "\n".join([json.dumps(r, ensure_ascii=False) for r in train_handwritten]), encoding="utf-8"
    )
    (DATASET_DIR / "val_unified.jsonl").write_text(
        "\n".join([json.dumps(r, ensure_ascii=False) for r in val_unified]), encoding="utf-8"
    )
    (DATASET_DIR / "test_holdout_pristine.jsonl").write_text(
        "\n".join([json.dumps(r, ensure_ascii=False) for r in test_pristine]), encoding="utf-8"
    )
    (DATASET_DIR / "test_holdout_noisy.jsonl").write_text(
        "\n".join([json.dumps(r, ensure_ascii=False) for r in test_noisy]), encoding="utf-8"
    )
    
    print("\n✅ Dataset splits generated:")
    print(f"   📄 Stage 1 (Printed Train): {len(train_printed)} samples")
    print(f"   ✍️ Stage 2 (Handwritten Train): {len(train_handwritten)} samples")
    print(f"   🧪 Validation Set: {len(val_unified)} samples")
    print(f"   🏆 Holdout Test (Pristine): {len(test_pristine)} samples ➔ dataset/test_holdout_pristine.jsonl")
    print(f"   🌪️ Holdout Test (Noisy): {len(test_noisy)} samples ➔ dataset/test_holdout_noisy.jsonl")

if __name__ == "__main__":
    build_staged_datasets()
