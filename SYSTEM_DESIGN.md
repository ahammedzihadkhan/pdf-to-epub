# System design: scanned Bangla PDF → EPUB

Runs fully offline on a CPU laptop (8 GB RAM, no GPU needed). Gemini is optional and only improves text quality.

## Pipeline

```mermaid
flowchart TD
    PDF[Scanned PDF] --> R[Stage 0: render pages to JPEG at 150 DPI]
    R --> C{Page 1?}
    C -- yes --> COVER[EPUB cover image]
    C -- no --> G{Gemini key?}
    G -- yes --> GO[Stage 1: Gemini OCR + photo boxes]
    G -- no --> L
    GO --> RC[Re-check: local layout model vs Gemini]
    L[Stage 2: local layout model finds photos] --> E[EasyOCR reads text with photos blanked out, finds chapter titles]
    GO --> F[Stage 3: crop photos]
    E --> TC[Learned text corrector] --> F
    F --> B[Stage 4: build EPUB, photos placed in reading order]
    COVER --> B
```

## Layout model training loop

```mermaid
flowchart LR
    GC[Gemini caches: photo boxes] --> S[Split 70/15/15 train/tune/test]
    S --> T[Train cell classifier]
    T --> K[Re-check training pages vs Gemini]
    K -- disagreements weight x2 --> T
    T --> RL[RL bandit picks post-processing on tune pages]
    RL --> EV[Compare with Gemini on test pages]
    EV --> M[layout_model.joblib + layout_report.json]
```

## ER diagram

Each entity is a file on disk. `<prefix>` is the book's cache prefix, such as `mahatirtha`.

```mermaid
erDiagram
    BOOK ||--|{ PAGE : has
    BOOK ||--|{ CHAPTER : "split into"
    CHAPTER ||--|{ PAGE : groups
    PAGE ||--o| GEMINI_OCR : "read by"
    PAGE ||--o| LOCAL_OCR : "read by"
    PAGE ||--o| LAYOUT_COMPARISON : "re-checked in"
    GEMINI_OCR ||--o{ PHOTO_BOX : lists
    LOCAL_OCR ||--o{ PHOTO_BOX : lists
    PHOTO_BOX ||--|| FIGURE : "cropped into"
    LAYOUT_MODEL ||--o{ LOCAL_OCR : "finds photos for"
    LAYOUT_MODEL ||--|| BANDIT_ARM : "tuned by"
    LAYOUT_MODEL ||--o{ RECHECK : "logs"
    GEMINI_OCR ||--o{ RECHECK : "labels"
    TEXT_CORRECTOR ||--o{ LOCAL_OCR : "fixes text of"
    GEMINI_OCR ||--o{ TEXT_CORRECTOR : "teaches"

    BOOK {
        string prefix PK "folder prefix"
        string pdf_path
        string title
        string author
        string epub_path
    }
    PAGE {
        string stem PK "page_0000, page_0000 is always the cover"
        string book_prefix FK
        string image_path "<prefix>_pages/page_NNNN.jpg"
        string format "blank | text | text+photo | photo"
    }
    GEMINI_OCR {
        string page_stem FK "<prefix>_ocr_cache/page_NNNN.json"
        string heading "major chapter title or null"
        string raw_text
        string content_markdown "has [PHOTO_n] markers"
        string footnotes
    }
    LOCAL_OCR {
        string page_stem FK "<prefix>_local_ocr_cache/page_NNNN.json"
        string heading "short isolated title row near the top, or null"
        string content_markdown "EasyOCR text + [PHOTO_n] markers"
        string page_format
    }
    PHOTO_BOX {
        string source "gemini | local"
        float_array box_2d "ymin xmin ymax xmax, 0-1000"
        string caption
    }
    FIGURE {
        string figure_id PK "page_NNNN_fig_NNN"
        string page FK
        string file "<prefix>_figures/*.jpg"
        int_array pixel_box
        string label "chitro N"
    }
    CHAPTER {
        string title "page heading; part N if a book has none"
        string file "chapter_NNN.xhtml inside the EPUB"
    }
    LAYOUT_MODEL {
        string path "bangla_ai_ocr/models/layout_model.joblib"
        object clf "gradient-boosted cell classifier"
        bool refine "gutter split on or off"
    }
    BANDIT_ARM {
        float threshold
        int min_cells
        int close_kernel
        float avg_reward "page F1 vs Gemini"
    }
    RECHECK {
        int round
        string page
        float f1
        string gemini_boxes
        string local_boxes
    }
    TEXT_CORRECTOR {
        string path "bangla_ai_ocr/models/text_corrector.json"
        object subs "misread word -> correct word"
        object vocab "correct word -> count"
        float test_cer "raw vs corrected, in text_corrector_report.json"
    }
    LAYOUT_COMPARISON {
        string page "<prefix>_dataset/layout_comparison.jsonl"
        string format
        float f1_vs_gemini
    }
```

## Low-resource choices

| Choice | Why |
|---|---|
| Layout model is a gradient-boosted classifier on 864 grid cells per page | Trains in minutes on 2 CPU threads and is under 1 MB |
| EasyOCR on CPU, loaded once, only for pages without Gemini | Peaks at about 1.6 GB RAM |
| `--threads 2` and `--sleep` | Keeps a hot laptop from throttling |
| Text corrector is a word map plus a vocabulary (JSON), learned from offline-vs-Gemini page pairs | Trains in seconds and adds no model to RAM; kept only if held-out CER drops |
| App preview reads page images and caches from disk once a second | The UI never waits on the OCR thread, and no extra copies sit in memory |
| Every stage caches to disk | Stop and resume at any time |
| TrOCR fine-tuning is not in the app | Its 334M-parameter model is too heavy for 8 GB RAM, and the old checkpoint had collapsed. The trainer's REINFORCE loss is now fixed, so retrain from the base model if you use it |

## Known limits

- Offline chapter titles come from a layout rule: a short, isolated row near the top of the page that is taller than the body text or starts low on the page. It was tuned on 11 pages from 3 books. If it finds no titles, the book is split into 20-page parts.
- Offline text accuracy is EasyOCR's. Gemini text is better when a key is available.
- The text corrector was trained on 60 pages and lowers offline error from 3.61% to 3.47% on 12 test pages. Some of its fixes copy Gemini's modern spelling (for example কখনও → কখনো) rather than the printed text. Delete `bangla_ai_ocr/models/text_corrector.json` to turn it off.
- The layout model has only seen three books. Re-train after adding Gemini caches for new books.
