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

# Bangla OCR Training Dataset: মহাতীর্থের শেষ যাত্রী

Extracted from 'মহাতীর্থের শেষ যাত্রী' by বিমল দে using Gemini Vision OCR pipeline.

## Fields
- id, source_book, author, language, ocr_model
- page_index, image_path
- heading, raw_text, content_markdown
- figures (list of {id, path, box, detector, label})
- has_figures

## Splits
| Split | Count | Ratio |
|-------|-------|-------|
| train | 403 | 90%   |
| val   | 45 | 10%   |
| all   | 448 | 100%  |
