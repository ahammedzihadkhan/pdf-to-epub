# 🇧🇩 Next-Gen AI-Powered Bangla OCR Pipeline

An advanced, end-to-end Optical Character Recognition (OCR) system for historical, printed, and complex Bengali documents combining state-of-the-art vision-language models, layout segmentation, and language model post-processing.

---

## 🔬 GitHub Research & Combined Technologies

To build a superior Bangla OCR model, we analyzed and combined key strengths from leading open-source repositories and research papers:

| Open Source Technology | Source Repository | Contribution to Pipeline |
|---|---|---|
| **CRAFT (Character Region Awareness)** | [`clovaai/CRAFT-pytorch`](https://github.com/clovaai/CRAFT-pytorch) | High-accuracy text bounding box & line region detection |
| **TrOCR (Vision-Language Transformer)** | [`microsoft/unilm/trocr`](https://github.com/microsoft/unilm/tree/master/trocr) & [`iitb-research-code/indic-trocr`](https://github.com/iitb-research-code/indic-trocr) | Vision Transformer (ViT) encoder + Autoregressive Decoder for complex Bengali ligatures |
| **BanglaBERT Language Model** | [`csebuetnlp/banglabert`](https://github.com/csebuetnlp/banglabert) | Post-OCR spell correction, contextual LM rescoring, and word error recovery |
| **EasyOCR Bengali CRNN** | [`JaidedAI/EasyOCR`](https://github.com/JaidedAI/EasyOCR) | Fast, lightweight character-level CTC recognition fallback |
| **BaDLAD Document Dataset** | [`BengaliAI/BaDLAD`](https://github.com/bengaliai/BaDLAD) | Document layout analysis conventions (margins, tables, headers) |

---

## 🏗️ Architecture Design

```
                     ┌───────────────────────────────┐
                     │ Scanned Bengali Document Page │
                     └───────────────┬───────────────┘
                                     │
                        [ CLAHE & Skew Correction ]
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │ Layout & Line Segmentation    │
                     │ (Morphological / CRAFT)       │
                     └───────────────┬───────────────┘
                                     │
                             (Line Image Crops)
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │ Bangla-TrOCR Model            │
                     │  - Vision Encoder: ViT        │
                     │  - Decoder: Bangla-RoBERTa    │
                     └───────────────┬───────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │ Script Purity & LM Post-Edit  │
                     │  - Remove mixed-script glyphs │
                     │  - Conjunct ligature repair   │
                     └───────────────┬───────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │ Output: Markdown / LaTeX / EPUB│
                     └───────────────────────────────┘
```

---

## 📦 Dataset Preparation (Using This Project's 616 Scanned Pages)

We utilize the 616 high-resolution scanned pages from *“বাংলাদেশের স্বাধীনতা: কূটনৈতিক যুদ্ধ”* along with their audited, high-purity ground-truth JSON transcriptions.

### To generate the training dataset:
```bash
python dataset_builder.py
```
This extracts:
- `dataset/images/*.jpg` — Thousands of cropped individual line images.
- `dataset/train.jsonl` — 85% training pairs `{"file_name": "images/line_0001.jpg", "text": "..."}`.
- `dataset/val.jsonl` — 15% validation evaluation set.

---

## 🏋️ Training the AI Model

To train / fine-tune the Bangla Vision-Encoder-Decoder model on your GPU/CPU:

```bash
python train_ocr.py
```
*Features:*
- Mixed precision training (`fp16`)
- Character Error Rate (**CER**) & Word Error Rate (**WER**) evaluation via `evaluate` and `jiwer`
- Checkpoint auto-saving to `models/bangla_ocr_checkpoint/`

---

## 🚀 Running Inference

```python
from hybrid_engine import HybridBanglaOCR

engine = HybridBanglaOCR()
text = engine.recognize("pages/page_0027.jpg")
print(text)
```
