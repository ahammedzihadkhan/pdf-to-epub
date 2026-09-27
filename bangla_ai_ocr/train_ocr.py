import os
import sys
import json
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

from .augmentations import OCRImageAugmentor

DATASET_DIR = Path(__file__).parent / "dataset"
OUTPUT_DIR = Path(__file__).parent / "models" / "bangla_ocr_checkpoint"

class BanglaOCRDataset(Dataset):
    def __init__(self, jsonl_file: Path, processor: TrOCRProcessor, max_target_length: int = 128, is_training: bool = False, force_noise: bool = False):
        self.data_dir = jsonl_file.parent
        self.processor = processor
        self.max_target_length = max_target_length
        self.is_training = is_training
        self.force_noise = force_noise
        self.augmentor = OCRImageAugmentor()
        self.records = []
        
        if jsonl_file.exists():
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.records.append(json.loads(line))
        print(f"Loaded {len(self.records)} samples from {jsonl_file.name} (Augment: {is_training or force_noise})")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        image_path = self.data_dir / item["file_name"]
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (384, 48), color="white")
            
        # Apply on-the-fly multi-resolution downscaling & noise injection
        if self.is_training or self.force_noise or item.get("is_corrupted", False):
            image = self.augmentor.augment(image)
            
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

def train_progressive():
    print("=== Progressive Bangla AI OCR Training with Noise Augmentations ===")
    
    model_name = "microsoft/trocr-base-handwritten"
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
    
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Datasets
    train_printed_ds = BanglaOCRDataset(DATASET_DIR / "train_printed.jsonl", processor, is_training=True)
    val_ds = BanglaOCRDataset(DATASET_DIR / "val_unified.jsonl", processor, is_training=False)

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        fp16=torch.cuda.is_available(),
        output_dir=str(OUTPUT_DIR),
        logging_steps=20,
        save_total_limit=2,
        num_train_epochs=3,
        learning_rate=5e-5,
        weight_decay=0.01,
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        compute_metrics=lambda p: compute_metrics(p, processor),
        train_dataset=train_printed_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator,
    )

    print("\nStarting Stage 1 Printed Training...")
    trainer.train()
    
    # Save model
    final_dir = OUTPUT_DIR / "final_model"
    model.save_pretrained(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"🎉 Final Model saved to: {final_dir}")

if __name__ == "__main__":
    train_progressive()
