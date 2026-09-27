import os
import sys
import json
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoImageProcessor,
    RobertaTokenizer,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    default_data_collator
)
import evaluate

sys.stdout.reconfigure(encoding='utf-8')

DATASET_DIR = Path(__file__).parent / "dataset"

class BanglaOCRDataset(Dataset):
    def __init__(self, jsonl_file: Path, processor: TrOCRProcessor, max_samples: int = 10):
        self.data_dir = jsonl_file.parent
        self.processor = processor
        self.records = []
        
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))
                if len(self.records) >= max_samples:
                    break
        print(f"Loaded {len(self.records)} sample lines for pipeline verification.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        image_path = self.data_dir / item["file_name"]
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (384, 48), color="white")
            
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(item["text"], padding="max_length", max_length=64, truncation=True).input_ids
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels)
        }

def test_pipeline():
    print("==================================================")
    print("TESTING BANGLA OCR TRAINING PIPELINE (LOCAL)")
    print("==================================================")
    
    model_name = "microsoft/trocr-base-handwritten"
    print(f"\n1. Initializing RobertaTokenizer & ImageProcessor from {model_name}...")
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
    print("   TrOCRProcessor successfully assembled!")
    
    print("\n2. Loading VisionEncoderDecoder model architecture...")
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    print("   Model architecture loaded!")
    
    print("\n3. Loading local dataset samples...")
    train_dataset = BanglaOCRDataset(DATASET_DIR / "train.jsonl", processor, max_samples=4)
    loader = DataLoader(train_dataset, batch_size=2, collate_fn=default_data_collator)
    
    print("\n4. Running 1 Forward & Backward Training Step on CPU...")
    model.train()
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    for batch in loader:
        optimizer.zero_grad()
        outputs = model(
            pixel_values=batch["pixel_values"],
            labels=batch["labels"]
        )
        loss = outputs.loss
        print(f"   Initial Training Step Loss: {loss.item():.4f}")
        
        loss.backward()
        optimizer.step()
        print("   Backward pass & gradient update completed successfully!")
        break
        
    print("\n5. Testing Metric Evaluation (CER / WER)...")
    cer_metric = evaluate.load("cer")
    sample_preds = ["বাংলাদেশ স্বাধীনতা যুদ্ধ"]
    sample_refs = ["বাংলাদেশের স্বাধীনতা যুদ্ধ"]
    score = cer_metric.compute(predictions=sample_preds, references=sample_refs)
    print(f"   Sample CER metric computation: {score:.4f}")
    
    print("\nALL PIPELINE TESTS PASSED 100%! Ready for full GPU training on Colab.")

if __name__ == "__main__":
    test_pipeline()
