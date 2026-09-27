import os
import sys
import json
import time
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    RobertaTokenizer,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator
)
import evaluate

sys.stdout.reconfigure(encoding='utf-8')

# Ensure multi-threading uses available CPU cores
torch.set_num_threads(6)

DATASET_DIR = Path(__file__).parent / "dataset"
OUTPUT_DIR = Path(__file__).parent / "models" / "laptop_trained_checkpoint"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class BanglaOCRDataset(Dataset):
    def __init__(self, jsonl_file: Path, processor: TrOCRProcessor, max_target_length: int = 128, max_samples: int = None):
        self.data_dir = jsonl_file.parent
        self.processor = processor
        self.max_target_length = max_target_length
        self.records = []
        
        if jsonl_file.exists():
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.records.append(json.loads(line))
                    if max_samples and len(self.records) >= max_samples:
                        break
        print(f"Loaded {len(self.records)} samples from {jsonl_file.name}")

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
        
        labels = self.processor.tokenizer(
            item["text"],
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True
        ).input_ids
        
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels)
        }

cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")

def compute_metrics(pred, processor):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer, "wer": wer}

def train_on_laptop():
    print("==================================================")
    print("💻 STARTING BANGLA OCR TRAINING ON LAPTOP (CPU)")
    print("==================================================")
    
    model_name = "microsoft/trocr-base-handwritten"
    print("\n1. Initializing Processor and Architecture...")
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
    
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Load datasets
    print("\n2. Loading Training & Validation Splits...")
    # Using the staged dataset
    train_dataset = BanglaOCRDataset(DATASET_DIR / "train_printed.jsonl", processor)
    eval_dataset = BanglaOCRDataset(DATASET_DIR / "val_unified.jsonl", processor, max_samples=50)

    # Configure CPU-optimized training arguments
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        per_device_train_batch_size=2,          # Small batch size for CPU RAM
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,          # Effective batch size = 8
        output_dir=str(OUTPUT_DIR),
        logging_steps=10,
        save_total_limit=2,
        num_train_epochs=2,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=30,
        report_to="none",
        dataloader_num_workers=0,               # Windows multiprocessing safety
        use_cpu=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        compute_metrics=lambda p: compute_metrics(p, processor),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    print("\n3. Starting Training on 8 CPU Cores...")
    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time
    print(f"\n🎉 Training Finished in {elapsed/60:.2f} minutes!")
    
    # Save final model
    final_dir = OUTPUT_DIR / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"📦 Final Model saved locally to: {final_dir}")

if __name__ == "__main__":
    train_on_laptop()
