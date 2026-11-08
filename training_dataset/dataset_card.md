---
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

# Bangla OCR Training Dataset

Extracted from a Bengali book using Gemini Vision OCR via manusher_pipeline.py.

## Fields
- id, source_book, author, language, ocr_model
- page_index, image_path
- heading, raw_text, content_markdown
- figures (list of {id, path, box, detector, label})
- has_figures

## Splits
| Split | Ratio |
|-------|-------|
| train | 90%   |
| val   | 10%   |
