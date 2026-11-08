"""
rl_trainer.py
─────────────
Reinforcement-Learning fine-tuning of the local Bangla TrOCR model.

Reward = CER-based alignment score vs Gemini 'gold standard' output.
  reward = 1.0 - CER(model_output, gemini_output)   (clipped to [-1, 1])

Training Algorithm: REINFORCE with a moving-average baseline.
  gradient = reward_advantage * log_prob(generated_tokens)

Dataset source: manusher_ocr_cache/*.json  (Gemini gold)
               manusher_pages/*.jpg         (page images)

The trainer:
  1. Loads your TrOCR model from checkpoint.
  2. For each page image, runs:
     a. Greedy decode with log-prob collection (for gradient).
     b. Scores output against Gemini gold using CER.
  3. Updates model weights using REINFORCE policy gradient.
  4. Saves improved checkpoint to bangla_ai_ocr/models/rl_finetuned/
  5. Writes a comparison report: rl_comparison_report.jsonl

Usage:
    python rl_trainer.py \
        --checkpoint bangla_ai_ocr/models/bangla_ocr_checkpoint \
        --pages_dir manusher_pages \
        --cache_dir manusher_ocr_cache \
        --epochs 2 \
        --lr 1e-5
"""

import sys
import os
import json
import argparse
import math
from pathlib import Path
from collections import deque

try:
    if sys.stdout is not None:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import torch
import torch.optim as optim
from PIL import Image

# ── Optional jiwer for CER ────────────────────────────────────────────────
try:
    from jiwer import cer as compute_cer
    _HAS_JIWER = True
except ImportError:
    _HAS_JIWER = False
    def compute_cer(ref, hyp):
        """Fallback character error rate."""
        if not ref:
            return 1.0
        ref, hyp = list(ref), list(hyp)
        # Simple Levenshtein character-level
        m, n = len(ref), len(hyp)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                if ref[i-1] == hyp[j-1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j-1])
                prev = temp
        return dp[n] / len(ref)


# ─────────────────────────────────────────────────────────────────────────────
def load_model(checkpoint_dir):
    from transformers import (
        AutoImageProcessor, RobertaTokenizer,
        TrOCRProcessor, VisionEncoderDecoderModel
    )
    ckpt_path = Path(checkpoint_dir)
    if ckpt_path.exists() and any(ckpt_path.iterdir()):
        model_source = str(ckpt_path)
    else:
        model_source = "microsoft/trocr-base-printed"
    print(f"Loading model from {model_source} ...")
    image_processor = AutoImageProcessor.from_pretrained(model_source)
    tokenizer = RobertaTokenizer.from_pretrained(model_source)
    processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
    model = VisionEncoderDecoderModel.from_pretrained(model_source)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.decoder_start_token_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"  Device: {device}")
    return model, processor, tokenizer, device


def get_page_pairs(pages_dir, cache_dir):
    """
    Returns list of (image_path, gemini_text) sorted by page number,
    skipping pages where Gemini cache is missing.
    """
    pages_dir = Path(pages_dir)
    cache_dir = Path(cache_dir)
    pairs = []
    for img_path in sorted(pages_dir.glob("page_*.jpg")):
        cache_file = cache_dir / f"{img_path.stem}.json"
        if not cache_file.exists():
            continue
        try:
            data = json.loads(cache_file.read_text(encoding='utf-8'))
            gold_text = data.get("content_markdown", "") or ""
            gold_text = gold_text.strip()
            if not gold_text:
                continue
        except Exception:
            continue
        pairs.append((img_path, gold_text))
    return pairs


def score_output(model_text, gold_text):
    """
    Reward = 1.0 - CER,  clipped to [0, 1].
    CER = 0  → reward = 1.0  (perfect match)
    CER >= 1 → reward = 0.0  (complete mismatch)
    """
    cer = compute_cer(gold_text, model_text)
    reward = max(0.0, 1.0 - cer)
    return reward, cer


def reinforce_step(model, processor, device, image_path, gold_text, optimizer, baseline):
    """
    Single REINFORCE step:
      1. Forward pass with teacher forcing to collect per-token log-probs.
      2. Greedy decode to get model_text for reward computation.
      3. Compute advantage = reward - baseline.
      4. Loss = -advantage * mean(log_probs).
      5. Backprop + gradient clip + optimizer step.
    Returns (model_text, reward, cer).
    """
    model.train()
    pil_img = Image.open(image_path).convert("RGB")
    pixel_values = processor(pil_img, return_tensors="pt").pixel_values.to(device)

    # ── Step 1: Greedy decode (no grad) to get model output text ──────────
    with torch.no_grad():
        gen_ids = model.generate(
            pixel_values,
            max_length=256,
            num_beams=1,     # greedy for speed during RL
            do_sample=False,
        )
    model_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

    # ── Step 2: Reward ────────────────────────────────────────────────────
    reward, cer = score_output(model_text, gold_text)
    advantage = reward - baseline

    # ── Step 3: log-prob of the sequence the model actually generated ────
    # generate() output starts with decoder_start_token; the model re-adds it when shifting labels.
    labels = gen_ids[:, 1:].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    optimizer.zero_grad()
    outputs = model(pixel_values=pixel_values, labels=labels)
    nll_gen = outputs.loss   # -log p(model_text | image), averaged over tokens

    # REINFORCE: loss = -advantage * log p(sample)
    # advantage > 0 → make this output more likely; advantage < 0 → less likely
    rl_loss = advantage * nll_gen
    rl_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return model_text, reward, cer


def run_rl_training(
    checkpoint_dir,
    pages_dir,
    cache_dir,
    output_dir,
    report_path,
    epochs=2,
    lr=1e-5,
    baseline_window=50,
    max_pages=None,
):
    model, processor, tokenizer, device = load_model(checkpoint_dir)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    pairs = get_page_pairs(pages_dir, cache_dir)
    if max_pages and max_pages > 0:
        pairs = pairs[:max_pages]
    if not pairs:
        print("No page-cache pairs found. Exiting RL training.")
        return

    print(f"RL Training on {len(pairs)} pages, {epochs} epoch(s), lr={lr}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = Path(report_path)
    done_pages = set()
    global_step = 0
    if report_path.exists():
        with open(str(report_path), "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        done_pages.add(rec.get("page"))
                        global_step = max(global_step, rec.get("step", 0))
                    except Exception:
                        pass

    report_file = open(str(report_path), "a", encoding="utf-8")
    if done_pages:
        print(f"  Resuming: {len(done_pages)} pages already trained. Skipping.")
        pairs = [p for p in pairs if p[0].stem not in done_pages]
        print(f"  Remaining pages to train: {len(pairs)}")

    reward_window = deque(maxlen=baseline_window)
    baseline = 0.5   # initial baseline estimate

    for epoch in range(1, epochs + 1):
        print(f"\n=== RL Epoch {epoch}/{epochs} ===")
        epoch_rewards = []

        for i, (img_path, gold_text) in enumerate(pairs):
            try:
                model_text, reward, cer = reinforce_step(
                    model, processor, device,
                    img_path, gold_text, optimizer, baseline
                )
            except Exception as e:
                print(f"  SKIP {img_path.name}: {e}")
                continue

            # Update moving-average baseline
            reward_window.append(reward)
            baseline = sum(reward_window) / len(reward_window)
            epoch_rewards.append(reward)
            global_step += 1

            record = {
                "step": global_step,
                "epoch": epoch,
                "page": img_path.stem,
                "gemini_gold": gold_text[:200],
                "model_output": model_text[:200],
                "cer": round(cer, 4),
                "reward": round(reward, 4),
                "baseline": round(baseline, 4),
            }
            report_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            report_file.flush()

            if global_step % 20 == 0:
                model.save_pretrained(str(output_dir))
                processor.save_pretrained(str(output_dir))

            if (i + 1) % 10 == 0 or i == len(pairs) - 1:
                avg_r = sum(epoch_rewards[-10:]) / len(epoch_rewards[-10:])
                print(f"  Step {global_step:4d} | page {img_path.stem} | "
                      f"CER={cer:.3f} | reward={reward:.3f} | "
                      f"avg_reward(10)={avg_r:.3f} | baseline={baseline:.3f}")

        avg_epoch_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0
        print(f"\nEpoch {epoch} complete — avg reward: {avg_epoch_reward:.4f}")

    report_file.close()

    # Save fine-tuned model
    print(f"\nSaving RL fine-tuned model to {output_dir} ...")
    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))
    print(f"RL Fine-tuned model saved to: {output_dir}")
    print(f"Comparison report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="RL Fine-tuning of Bangla TrOCR using Gemini as gold standard.")
    parser.add_argument("--checkpoint", type=str,
                        default="bangla_ai_ocr/models/bangla_ocr_checkpoint",
                        help="Path to trained TrOCR checkpoint directory")
    parser.add_argument("--pages_dir", type=str,
                        default="manusher_pages",
                        help="Directory with page JPEGs")
    parser.add_argument("--cache_dir", type=str,
                        default="manusher_ocr_cache",
                        help="Directory with Gemini OCR JSON cache files")
    parser.add_argument("--output_dir", type=str,
                        default="bangla_ai_ocr/models/rl_finetuned",
                        help="Where to save the RL fine-tuned model")
    parser.add_argument("--report", type=str,
                        default="manusher_dataset/rl_comparison_report.jsonl",
                        help="Path to write per-page comparison report")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_pages", type=int, default=None, help="Max pages to train on")
    parser.add_argument("--threads", type=int, default=2, help="CPU threads to keep laptop cool")
    args = parser.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    run_rl_training(
        checkpoint_dir=Path(args.checkpoint),
        pages_dir=Path(args.pages_dir),
        cache_dir=Path(args.cache_dir),
        output_dir=Path(args.output_dir),
        report_path=Path(args.report),
        epochs=args.epochs,
        lr=args.lr,
        max_pages=args.max_pages,
    )


if __name__ == "__main__":
    main()
