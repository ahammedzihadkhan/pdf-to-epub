"""
bangla_pdf_to_epub.py
=====================
Scanned Bangla PDF -> EPUB. Runs offline on a low-resource CPU laptop; Gemini is optional.

STAGES:
  0. Render PDF pages to JPEG (cached). Page 1 is ALWAYS the EPUB cover.
  1. Gemini Vision OCR, only if an API key is given (cached, resumable).
  2. Local layout model finds photo / text / blank regions on every page.
     - pages with Gemini: re-check local model vs Gemini -> <prefix>_dataset/layout_comparison.jsonl
     - pages without Gemini: EasyOCR (offline) reads text with photo regions blanked out
  3. Crop photos into <prefix>_figures/.
  4. Build the EPUB (verse lines kept, photos placed in reading order).

USAGE:
    python -m bangla_ai_ocr.layout_model train           # once: train the layout model
    python bangla_pdf_to_epub.py book.pdf --offline --sleep 0.5
    python bangla_pdf_to_epub.py "মহাতীর্থের শেষ যাত্রী ।। বিমল দে.pdf" --prefix mahatirtha   # reuse old caches
"""

import sys
import os
import json
import time
import argparse
import concurrent.futures
import re
import html
import statistics
from pathlib import Path

# ── Unicode stdout safety ─────────────────────────────────────────────────
try:
    if sys.stdout is not None:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import cv2
import fitz  # PyMuPDF
from PIL import Image
from tqdm import tqdm

import ebooklib
from ebooklib import epub
import markdown as md_lib

from google import genai
from google.genai import types

# ── Paths & configuration ─────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

# Set per book by configure(); folders follow the <prefix>_pages / <prefix>_ocr_cache convention
PDF_PATH = OUTPUT_EPUB = PAGES_DIR = FIGURES_DIR = OCR_CACHE = LOCAL_CACHE = DATASET_DIR = LAYOUT_REPORT = None
BOOK_TITLE = BOOK_AUTHOR = ""


def configure(pdf_path, out_epub=None, prefix=None, title=None, author=""):
    global PDF_PATH, OUTPUT_EPUB, PAGES_DIR, FIGURES_DIR, OCR_CACHE, LOCAL_CACHE, DATASET_DIR
    global LAYOUT_REPORT, BOOK_TITLE, BOOK_AUTHOR, _RUNNING_HEADER_NOISE
    PDF_PATH = Path(pdf_path)
    OUTPUT_EPUB = Path(out_epub) if out_epub else PDF_PATH.with_suffix(".epub")
    prefix = prefix or re.sub(r'[\\/:*?"<>|।\s]+', '_', PDF_PATH.stem).strip('_')[:60] or "book"
    PAGES_DIR = BASE_DIR / f"{prefix}_pages"
    FIGURES_DIR = BASE_DIR / f"{prefix}_figures"
    OCR_CACHE = BASE_DIR / f"{prefix}_ocr_cache"
    LOCAL_CACHE = BASE_DIR / f"{prefix}_local_ocr_cache"
    DATASET_DIR = BASE_DIR / f"{prefix}_dataset"
    LAYOUT_REPORT = DATASET_DIR / "layout_comparison.jsonl"
    BOOK_TITLE = title or PDF_PATH.stem
    BOOK_AUTHOR = author or ""
    noise = "|".join(re.escape(x) for x in (BOOK_TITLE, BOOK_AUTHOR) if x)
    _RUNNING_HEADER_NOISE = re.compile(rf'^({noise})\s*$', re.UNICODE)
    for d in [PAGES_DIR, FIGURES_DIR, OCR_CACHE, LOCAL_CACHE, DATASET_DIR]:
        d.mkdir(parents=True, exist_ok=True)

# ── Gemini configuration ──────────────────────────────────────────────────
def get_api_key():
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    env_file = BASE_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding='utf-8', errors='ignore').splitlines():
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip()
    return ""

GEMINI_MODELS = [
    "gemini-3.5-flash-lite",   # primary — high quota
    "gemini-3.5-flash",        # fallback 1
    "gemini-3.7-flash",        # fallback 2
    "gemini-3.8-flash",        # fallback 3
]

OCR_PROMPT = """\
You are an expert transcriber and EPUB formatter for Bengali books.
Transcribe this scanned page from a Bengali book.

RULES:
1. RUNNING HEADERS — SKIP COMPLETELY:
   Many pages have a running header at the very top: either the book title
   or the chapter name repeated. These are NOT part of the
   body text. DO NOT include them in content_markdown.

2. RUNNING FOOTERS — SKIP COMPLETELY:
   Many pages have a short text label at the very bottom, inside a box or
   underline, that repeats the chapter or section name (e.g. 'গল্পের শুরু',
   'মানুষ হিসেবে আমাদের অপরাধসমূহ'). These are running footers, NOT body
   text. DO NOT include them in content_markdown.

3. PAGE NUMBER FOOTERS — SKIP COMPLETELY:
   The very last element on most pages is a lone page number (e.g. ৩১, 55).
   Footnote references (small superscript numbers or text after a dash/line at
   the bottom) are also footer content. DO NOT include page numbers or
   footnote markers in content_markdown. Capture the actual footnote body text
   in the "footnotes" field instead.

4. Transcribe ALL Bengali and English body text verbatim. Fix minor OCR
   mistakes but keep the original tone, style and wording intact.

5. POETRY / VERSE: Preserve EACH verse line on its OWN separate line.
   NEVER merge verse lines into a single paragraph.

6. HEADINGS:
   - "heading" field: set ONLY when this page opens a new MAJOR chapter
     (e.g. 'মাটি ও মানুষ', 'খেলারাম খেলে যা'). Sub-sections like
     'কিস্তি : ১', 'কিস্তি : ২', 'পর্ব : ২' are NOT new chapters — put
     them in content_markdown as bold text **কিস্তি : ১**.
   - In content_markdown, do NOT use Markdown heading markers (# ## ###).

7. REAL PHOTOS / ILLUSTRATIONS ONLY:
   - Only flag a region as a photo if it is a REAL PHOTOGRAPH or hand-drawn
     ILLUSTRATION — i.e. content that cannot be typed as text.
   - DO NOT flag: decorative text boxes, bordered section labels, ruled lines,
     dividers, running footers, or any region that contains readable Bengali/English text.
   - For real images: insert [PHOTO_0], [PHOTO_1] in content_markdown, provide
     bounding box in "photos" as [ymin, xmin, ymax, xmax] normalized 0-1000,
     and a short Bengali caption.

8. Return ONLY valid JSON — no markdown fences, no extra keys:
{
  "heading": "Major chapter title string, or null",
  "raw_text": "Raw Bengali text exactly as printed (before correction)",
  "content_markdown": "Body text only — no running header, no running footer, no page number, no footnote markers. Verse lines on separate lines. Sub-sections as **bold** text. [PHOTO_N] for real images only.",
  "footnotes": "Footnote body text from bottom of page, or null",
  "photos": [
    {
      "box_2d": [ymin, xmin, ymax, xmax],
      "caption": "Bengali caption — real photographs/illustrations only"
    }
  ]
}"""


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 0: PDF → Images
# ═════════════════════════════════════════════════════════════════════════════
def pdf_to_images(pdf_path: Path, dpi: int = 150):
    print(f"\n[STAGE 0] Extracting pages from: {pdf_path.name}")
    doc = fitz.open(str(pdf_path))
    page_data = []
    new_count = 0
    for i in tqdm(range(len(doc)), desc="PDF -> Images"):
        page = doc[i]
        img_path = PAGES_DIR / f"page_{i:04d}.jpg"
        if not img_path.exists():
            pix = page.get_pixmap(dpi=dpi)
            pix.save(str(img_path))
            new_count += 1
        page_data.append({"img_path": img_path, "page_num": i})
    print(f"  {len(page_data)} pages total, {new_count} newly extracted.")
    return page_data


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 1: Gemini OCR
# ═════════════════════════════════════════════════════════════════════════════
def gemini_ocr_page(client, image_path: Path):
    """OCR a single page with Gemini. Returns parsed JSON dict or None."""
    cache_file = OCR_CACHE / f"{image_path.stem}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding='utf-8'))
        except Exception:
            pass  # re-process if cache is corrupt

    img_bytes = image_path.read_bytes()
    for model_name in GEMINI_MODELS:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                        OCR_PROMPT,
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1,
                    )
                )
                data = json.loads(response.text.strip())
                cache_file.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
                # Append to training datasets
                raw = data.get("raw_text", "")
                corrected = data.get("content_markdown", "")
                if raw and corrected:
                    ocr_ds = DATASET_DIR / "ocr_dataset.jsonl"
                    spell_ds = DATASET_DIR / "spell_correction_dataset.jsonl"
                    with open(ocr_ds, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"image": image_path.name, "text": raw},
                                           ensure_ascii=False) + "\n")
                    with open(spell_ds, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"bad_text": raw, "corrected_text": corrected},
                                           ensure_ascii=False) + "\n")
                return data
            except Exception as e:
                print(f"  [{model_name}] error on {image_path.name} attempt {attempt+1}: {e}")
                time.sleep(2.0)
    return None


def run_gemini_ocr(page_data, api_key, workers=2, progress=None, checkpoint=None):
    print(f"\n[STAGE 1] Gemini OCR")
    if not api_key:
        print("  ERROR: No Gemini API key found. Set GEMINI_API_KEY in .env or environment.")
        return
    client = genai.Client(api_key=api_key)
    to_process = [p for p in page_data[1:]  # page 1 is the cover: never sent to Gemini
                  if not (OCR_CACHE / f"{p['img_path'].stem}.json").exists()]
    if not to_process:
        print(f"  All {len(page_data) - 1} body pages already cached. Skipping.")
        return
    print(f"  Processing {len(to_process)} pages with {workers} parallel workers...")
    completed = 0

    def job(path):
        if checkpoint:
            checkpoint()  # runs in the worker, so pausing really pauses the Gemini calls
        return gemini_ocr_page(client, path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(job, p['img_path']): p for p in to_process}
        try:
            for future in concurrent.futures.as_completed(futures):
                p = futures[future]
                completed += 1
                result = future.result()
                status = "OK" if result else "FAIL"
                if progress:
                    progress(completed, len(to_process))
                if completed % 5 == 0 or completed == len(to_process):
                    print(f"  [{completed}/{len(to_process)}] {p['img_path'].name} -> {status}")
        except BaseException:
            executor.shutdown(cancel_futures=True)  # on stop/error, drop queued pages instead of finishing them
            raise


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 2: Local layout model (photo / text / blank) + offline EasyOCR text
# ═════════════════════════════════════════════════════════════════════════════
# Offline chapter-title rule, tuned on 11 + checked on 21 pages of 3 books against Gemini headings:
# a short, isolated row near the top that is taller than body text or starts low on the page.
HEAD_SKIP, HEAD_WIDTH, HEAD_GAP, HEAD_RATIO, HEAD_LOW, HEAD_MAX_CHARS = 2, 0.7, 0.6, 1.2, 0.15, 60


def _split_heading(lines, page_h, low_start_ok=True):
    """lines: EasyOCR (box, text, conf). Return (heading text or None, lines without the heading)."""
    if len(lines) < 4:
        return None, lines  # too few lines to know the body text size
    top = [min(p[1] for p in b) for b, _, _ in lines]
    bot = [max(p[1] for p in b) for b, _, _ in lines]
    x0 = [min(p[0] for p in b) for b, _, _ in lines]
    x1 = [max(p[0] for p in b) for b, _, _ in lines]
    body = statistics.median(b - t for t, b in zip(top, bot)) or 1
    text_w = (max(x1) - min(x0)) or 1
    order = sorted(range(len(lines)), key=top.__getitem__)
    for i in order[:HEAD_SKIP + 1]:  # a running header / page number may sit above the title
        row = [j for j in order if abs(top[j] - top[i]) < 0.5 * body]  # EasyOCR may split one line into boxes
        r_top, r_bot = min(top[j] for j in row), max(bot[j] for j in row)
        below = [top[j] for j in order if top[j] > r_bot - 0.3 * body]
        gap = ((min(below) if below else page_h) - r_bot) / body
        narrow = (max(x1[j] for j in row) - min(x0[j] for j in row)) / text_w < HEAD_WIDTH
        tall = (r_bot - r_top) / body >= HEAD_RATIO
        low = low_start_ok and r_top >= HEAD_LOW * page_h
        text = " ".join(lines[j][1] for j in sorted(row, key=x0.__getitem__)).strip()
        if narrow and gap >= HEAD_GAP and (tall or low) and len(text) <= HEAD_MAX_CHARS:
            return text, [l for j, l in enumerate(lines) if j not in row]
    return None, lines


def _local_page_json(engine, img_bgr, photos, fix=None):
    """EasyOCR the page with photo regions blanked out; [PHOTO_i] placed in reading order.
    fix: optional learned text corrector (bangla_ai_ocr/text_corrector.py), applied after the rule-based one."""
    fix = fix or (lambda s: s)
    from easyocr.utils import get_paragraph
    h, w = img_bgr.shape[:2]
    prep = engine.preprocess_image(img_bgr)
    for y0, x0, y1, x1 in photos:
        cv2.rectangle(prep, (int(x0 * w / 1000), int(y0 * h / 1000)), (int(x1 * w / 1000), int(y1 * h / 1000)), 255, -1)
    # same as readtext(paragraph=True), but line heights are still known for the heading rule
    heading, lines = _split_heading(engine.easyocr_reader.readtext(prep, detail=1, paragraph=False), h,
                                    low_start_ok=not photos)  # text under a photo starts low too
    items = [(min(pt[1] for pt in box), html.escape(fix(engine.corrector.correct_text(text))))
             for box, text in (get_paragraph(lines, x_ths=1, y_ths=0.5) if lines else [])]
    items += [(b[0] * h / 1000, f"[PHOTO_{i}]") for i, b in enumerate(photos)]
    return {
        "heading": fix(engine.corrector.correct_text(heading)) if heading else None,
        "content_markdown": "\n\n".join(t for _, t in sorted(items, key=lambda it: it[0]) if t.strip()),
        "footnotes": None,
        "photos": [{"box_2d": b, "caption": "", "source": "local"} for b in photos],
    }


def run_layout_stage(page_data, offline_text=True, threads=2, sleep=0.0, progress=None, checkpoint=None):
    print("\n[STAGE 2] Local layout model + offline OCR")
    sys.path.insert(0, str(BASE_DIR))
    from bangla_ai_ocr.layout_model import LayoutModel, MODEL_PATH, norm_box, page_f1, imread
    from bangla_ai_ocr.text_corrector import TextCorrector, MODEL_PATH as CORRECTOR_PATH

    model = LayoutModel() if MODEL_PATH.exists() else None
    if model is None:
        print("  No layout model: photos only come from Gemini. Train: python -m bangla_ai_ocr.layout_model train")
    fix = TextCorrector().fix if CORRECTOR_PATH.exists() else None  # trained: python -m bangla_ai_ocr.text_corrector train
    engine = None
    body = page_data[1:]  # page 1 is always the cover, never OCR'd
    with open(LAYOUT_REPORT, "w", encoding="utf-8") as rep:
        for k, p in enumerate(body, 1):
            if checkpoint:
                checkpoint()  # app pause/stop
            stem = p['img_path'].stem
            gem, loc = OCR_CACHE / f"{stem}.json", LOCAL_CACHE / f"{stem}.json"
            if gem.exists() and model:
                # Re-check: compare the local layout model against Gemini on this page
                try:
                    photos = json.loads(gem.read_text(encoding='utf-8')).get("photos") or []
                    gold = [b for b in (norm_box((ph or {}).get("box_2d")) for ph in photos) if b]
                    res = model.analyze(imread(p['img_path']))
                    rep.write(json.dumps({"page": stem, "format": res["format"],
                                          "f1_vs_gemini": round(page_f1(res["photos"], gold), 3),
                                          "gemini": gold, "local": res["photos"]}) + "\n")
                except Exception as e:
                    print(f"  re-check skipped [{stem}]: {e}")
            elif not gem.exists() and not loc.exists() and offline_text:
                img = imread(p['img_path'])
                if img is None:
                    raise RuntimeError(f"Cannot read page image {p['img_path']}")
                res = model.analyze(img) if model else {"format": "unknown", "photos": []}
                if engine is None:
                    import torch
                    torch.set_num_threads(threads)
                    from bangla_ai_ocr.hybrid_engine import HybridBanglaOCR
                    engine = HybridBanglaOCR()
                data = _local_page_json(engine, img, res["photos"], fix)
                data["page_format"] = res["format"]
                loc.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"  [{k}/{len(body)}] {stem}: {res['format']}, {len(res['photos'])} photo(s)")
                if sleep:
                    time.sleep(sleep)  # keep a low-power CPU cool
            if progress:
                progress(k, len(body))


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 3: Figure detection & caching
# ═════════════════════════════════════════════════════════════════════════════
def run_figure_detection(page_data):
    print(f"\n[STAGE 3] Figure / Photo cropping (Gemini boxes, else local layout model)")
    sys.path.insert(0, str(BASE_DIR))
    from bangla_ai_ocr.figure_detector import (
        detect_and_cache_figures, save_figures_index, load_figures_index
    )

    figures_index = load_figures_index(FIGURES_DIR)
    already_done = {r["page"] for r in figures_index}

    new_pages = [p for p in page_data if p['img_path'].stem not in already_done]
    if not new_pages:
        print(f"  All {len(page_data)} pages already have figures cached ({len(figures_index)} figures).")
        return figures_index

    print(f"  Scanning {len(new_pages)} pages for figures...")
    for p in tqdm(new_pages, desc="Figure detection"):
        cache_file = OCR_CACHE / f"{p['img_path'].stem}.json"
        if not cache_file.exists():
            cache_file = LOCAL_CACHE / f"{p['img_path'].stem}.json"
        detect_and_cache_figures(
            image_path=p['img_path'],
            cache_file=cache_file,
            figures_dir=FIGURES_DIR,
            figures_index=figures_index,
        )

    save_figures_index(FIGURES_DIR, figures_index)
    print(f"  Total figures found and cached: {len(figures_index)}")
    return figures_index


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 6: EPUB Assembly
# ═════════════════════════════════════════════════════════════════════════════

# CSS for the EPUB — handles both prose and verse
EPUB_CSS = """\
body {
    font-family: "Noto Serif Bengali", "Kalpurush", serif;
    line-height: 1.8;
    margin: 2em 1.5em;
    color: #1a1a1a;
}
h1, h2 {
    text-align: center;
    margin: 2em 0 1em 0;
    color: #2c2c2c;
}
p {
    text-align: justify;
    margin: 0.6em 0;
    text-indent: 1.5em;
}
p.verse {
    font-style: italic;
    text-align: left;
    text-indent: 0;
    margin: 0.2em 0 0.2em 2em;
    white-space: pre-wrap;
}
figure {
    text-align: center;
    margin: 1.5em auto;
    page-break-inside: avoid;
}
figure img {
    max-width: 100%;
    height: auto;
}
figcaption {
    font-size: 0.85em;
    color: #555;
    margin-top: 0.4em;
}
p.footnote {
    font-size: 0.80em;
    color: #666;
    border-top: 1px solid #ccc;
    padding-top: 0.4em;
    margin-top: 1em;
    text-indent: 0;
    text-align: left;
}
"""

# Threshold: lines with fewer chars than this in a paragraph are treated as verse lines
VERSE_LINE_THRESHOLD = 45


def _is_verse_line(line: str) -> bool:
    """
    Heuristic: a line is verse if it is short and doesn't end
    with a full-stop danda (।) followed by more text.
    """
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) < VERSE_LINE_THRESHOLD:
        return True
    return False


# Regex to detect Markdown heading lines that Gemini occasionally outputs
_MD_HEADING_RE = re.compile(r'^(#{1,6})\s+(.*)')


def _render_markdown_block(text: str) -> str:
    """
    Convert a block of markdown text to HTML, preserving verse lines.

    Handles:
    - Markdown headings (# ## ### → <h1>/<h2>/<h3>) — Gemini sometimes
      puts '### কিস্তি : ৩' in content_markdown even though we ask it
      not to. We convert them properly instead of letting them appear
      as raw '###' characters in the EPUB.
    - Verse lines → <p class="verse">
    - Normal prose paragraphs → <p>
    """
    if not text.strip():
        return ""

    html_parts = []
    raw_paragraphs = re.split(r'\n{2,}', text.strip())

    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue

        # Pass figure HTML through as-is
        if para.startswith("<figure"):
            html_parts.append(para)
            continue

        lines = para.split('\n')

        # ── Markdown heading lines ──────────────────────────────────────
        # If the *entire* paragraph is a heading line (Gemini artefact)
        if len(lines) == 1:
            m = _MD_HEADING_RE.match(lines[0].strip())
            if m:
                level = min(len(m.group(1)), 3)        # cap at h3
                tag   = f"h{level + 1}"               # h1 reserved for chapter; use h2/h3/h4
                content = m.group(2).strip()
                # strip trailing **bold** wrapper if heading is already bold
                content = re.sub(r'^\*\*(.+?)\*\*$', r'\1', content)
                html_parts.append(f'<{tag}>{content}</{tag}>')
                continue

        # ── Verse block ─────────────────────────────────────────────────
        if len(lines) > 1 and all(_is_verse_line(l) for l in lines if l.strip()):
            for line in lines:
                if line.strip():
                    html_parts.append(f'<p class="verse">{line.strip()}</p>')
        elif len(lines) == 1 and _is_verse_line(lines[0]):
            single = lines[0].strip()
            m = _MD_HEADING_RE.match(single)
            if m:
                level   = min(len(m.group(1)), 3)
                tag     = f"h{level + 1}"
                content = re.sub(r'^\*\*(.+?)\*\*$', r'\1', m.group(2).strip())
                html_parts.append(f'<{tag}>{content}</{tag}>')
            else:
                html_parts.append(f'<p class="verse">{single}</p>')
        else:
            # ── Prose paragraph ─────────────────────────────────────────
            # Check individual lines for embedded headings before joining
            processed_lines = []
            for line in lines:
                m = _MD_HEADING_RE.match(line.strip())
                clean_line = re.sub(r'^\*+|\*+$', '', line.strip())
                is_sub_sec = bool(_SUB_SECTION_RE.match(clean_line)) if clean_line else False
                
                if m or is_sub_sec:
                    # flush accumulated prose first
                    if processed_lines:
                        prose = " ".join(l.strip() for l in processed_lines if l.strip())
                        prose = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', prose)
                        prose = re.sub(r'\*(.+?)\*', r'<em>\1</em>', prose)
                        html_parts.append(f'<p>{prose}</p>')
                        processed_lines = []
                    
                    if m:
                        level   = min(len(m.group(1)), 3)
                        tag     = f"h{level + 1}"
                        content = re.sub(r'^\*\*(.+?)\*\*$', r'\1', m.group(2).strip())
                    else:
                        tag = "h3"
                        content = clean_line
                    html_parts.append(f'<{tag}>{content}</{tag}>')
                else:
                    processed_lines.append(line)
            if processed_lines:
                prose = " ".join(l.strip() for l in processed_lines if l.strip())
                prose = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', prose)
                prose = re.sub(r'\*(.+?)\*', r'<em>\1</em>', prose)
                html_parts.append(f'<p>{prose}</p>')

    return "\n".join(html_parts)


# ── Chapter-classification helpers ──────────────────────────────────────────
# Sub-section patterns that look like headings but belong INSIDE a chapter
_SUB_SECTION_RE = re.compile(
    r'^(কিস্তি|পর্ব|অধ্যায়|ভাগ|অংশ|কস্তি|কিস্টি|কিত্তি)\s*[:\-–]?\s*[\d০-৯]+\.?$',
    re.UNICODE
)
# Patterns for running page-header noise (chapter name repeated at top)
_RUNNING_HEADER_NOISE = re.compile(r'(?!x)x')  # replaced by configure() with the book title/author
# Lone page-number line: Bengali or ASCII digits only
_PAGE_NUMBER_RE = re.compile(r'^[\d০-৯]+\.?\s*$')


def _is_true_chapter(heading: str) -> bool:
    """Return True if heading is a MAJOR chapter, False if it is a sub-section."""
    if not heading:
        return False
    h = heading.strip()
    if _SUB_SECTION_RE.match(h):
        return False
    if _RUNNING_HEADER_NOISE.match(h):
        return False
    return True


def _clean_content(md: str, chapter_title: str,
                   all_chapter_titles: set | None = None) -> str:
    """
    Strip artefacts from Gemini content_markdown:
    - Running page header (chapter name / book title at top)
    - Running page footer (chapter name repeated at bottom in a label box)
    - Bare page-number line at bottom
    - Stray Markdown # heading prefixes for the chapter title itself
    - Stray HTML tags Gemini occasionally outputs (e.g. </div>)

    all_chapter_titles: set of ALL known chapter titles across the book,
    so we can match running footers even when they differ from the
    current chapter (e.g. the footer shows the PREVIOUS chapter name).
    """
    if not md:
        return ""

    # Remove any stray HTML tags Gemini sometimes outputs
    md = re.sub(r'</?div[^>]*>', '', md)
    md = re.sub(r'</?span[^>]*>', '', md)

    lines = md.splitlines()
    known_titles = all_chapter_titles or set()

    def _is_running_noise(text: str) -> bool:
        """True if `text` is a running header/footer to remove."""
        t = text.strip()
        t_clean = re.sub(r'^#{1,6}\s*', '', t).strip()   # strip any # prefix
        t_clean = re.sub(r'^\*+|\*+$', '', t_clean).strip()  # strip bold markers
        if not t_clean:
            return True   # blank line
        if _RUNNING_HEADER_NOISE.match(t):
            return True
        if _RUNNING_HEADER_NOISE.match(t_clean):
            return True
        # Match current chapter title
        if chapter_title and t_clean == chapter_title.strip():
            return True
        if chapter_title and t == chapter_title.strip():
            return True
        # Match any known chapter title (catches running footers from other chapters)
        if t_clean in known_titles:
            return True
        if t in known_titles:
            return True
        return False

    # Strip leading running header lines (up to 3 lines)
    stripped_top = 0
    while lines and stripped_top < 3:
        if _is_running_noise(lines[0]):
            lines.pop(0)
            stripped_top += 1
        else:
            break

    # Strip trailing running footer / page-number lines (up to 3 lines)
    stripped_btm = 0
    while lines and stripped_btm < 3:
        last = lines[-1].strip()
        last_clean = re.sub(r'^\*+|\*+$', '', last).strip()
        if not last:
            lines.pop()
            stripped_btm += 1
        elif _PAGE_NUMBER_RE.match(last):
            lines.pop()
            stripped_btm += 1
        elif _is_running_noise(last):
            lines.pop()
            stripped_btm += 1
        elif _is_running_noise(last_clean):
            lines.pop()
            stripped_btm += 1
        else:
            break

    return '\n'.join(lines)


def build_epub(page_data, figures_index: list, output_file: Path):
    print(f"\n[STAGE 6] Building EPUB: {output_file.name}")
    print("  Pass 1: collecting all page OCR data...")

    # ── Pass 1: load all cached OCR data ─────────────────────────────────────
    all_pages = []   # list of (page_stem, heading, content_md, photos, footnotes)
    for p in page_data:
        cache_file = OCR_CACHE / f"{p['img_path'].stem}.json"
        if not cache_file.exists():
            cache_file = LOCAL_CACHE / f"{p['img_path'].stem}.json"
        if not cache_file.exists():
            continue
        try:
            data = json.loads(cache_file.read_text(encoding='utf-8'))
        except Exception:
            continue
        all_pages.append({
            "stem":      p['img_path'].stem,
            "heading":   (data.get("heading") or "").strip(),
            "content":   data.get("content_markdown", "") or "",
            "photos":    data.get("photos", []),
            "footnotes": data.get("footnotes") or "",
        })

    print(f"  {len(all_pages)} pages loaded.")

    # Separate cover page (page_0000) from body pages
    cover_page_img = PAGES_DIR / "page_0000.jpg"
    # Skip page_0000 from body chapters — it will be the EPUB cover
    body_pages = [p for p in all_pages if p["stem"] != "page_0000"]

    # ── Collect ALL known chapter titles for footer matching ──────────────────
    all_chapter_titles: set = set()
    for page in all_pages:
        h = page["heading"]
        if h and h.lower() not in ("null", "none"):
            all_chapter_titles.add(h.strip())
    # Also add book/author noise
    all_chapter_titles.update(x for x in (BOOK_TITLE, BOOK_AUTHOR) if x)
    print(f"  {len(all_chapter_titles)} chapter/footer labels identified for stripping.")

    # ── Pass 2: identify TRUE chapter boundaries ──────────────────────────────
    # Group consecutive pages under the same major chapter.
    # Sub-sections (কিস্তি etc.) become bold headings within the chapter body.
    chapter_groups = []  # list of {"title": str, "pages": [page_dict, ...]}
    has_headings = any(p["heading"] and _is_true_chapter(p["heading"]) for p in body_pages)
    bn_num = lambda n: str(n).translate(str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯"))
    current_chapter = {"title": "ভূমিকা" if has_headings else "অংশ ১", "pages": []}

    for page in body_pages:
        h = page["heading"]
        if h and _is_true_chapter(h):
            # Save current chapter if it has content
            if current_chapter["pages"]:
                chapter_groups.append(current_chapter)
            current_chapter = {"title": h, "pages": [page]}
        else:
            # shortcut: no chapter titles found (the offline rule missed them), so cut into 20-page parts
            if not has_headings and len(current_chapter["pages"]) >= 20:
                chapter_groups.append(current_chapter)
                current_chapter = {"title": f"অংশ {bn_num(len(chapter_groups) + 1)}", "pages": []}
            current_chapter["pages"].append(page)

    if current_chapter["pages"]:
        chapter_groups.append(current_chapter)

    print(f"  {len(chapter_groups)} chapters identified.")

    # ── Build EPUB book object ────────────────────────────────────────────────
    fig_by_page: dict = {}
    for fig in figures_index:
        fig_by_page.setdefault(fig["page"], []).append(fig)

    book = epub.EpubBook()
    book.set_identifier(PAGES_DIR.name)
    book.set_title(BOOK_TITLE)
    book.set_language("bn")
    if BOOK_AUTHOR:
        book.add_author(BOOK_AUTHOR)

    css_item = epub.EpubItem(
        uid="main_css",
        file_name="style/main.css",
        media_type="text/css",
        content=EPUB_CSS
    )
    book.add_item(css_item)

    chapters      = []
    chapter_index = 1
    book_images_added = set()

    # ── Cover image (page_0000 = front cover, full page) ─────────────────────
    has_cover = False
    if cover_page_img.exists():
        cover_img_bytes = cover_page_img.read_bytes()
        book_images_added.add("cover_image")
        # set_cover() registers image, cover metadata AND cover.xhtml in one call
        book.set_cover("images/cover.jpg", cover_img_bytes)
        has_cover = True
        print("  Cover page added from page_0000.jpg")

    def _add_figure_to_epub(fig_record):
        fig_path      = FIGURES_DIR / fig_record["file"]
        uid           = fig_record["figure_id"]
        epub_img_path = f"images/{fig_record['file']}"
        if uid not in book_images_added and fig_path.exists():
            book.add_item(epub.EpubItem(
                uid=uid,
                file_name=epub_img_path,
                media_type="image/jpeg",
                content=fig_path.read_bytes()
            ))
            book_images_added.add(uid)
        return epub_img_path

    # ── Pass 3: render each chapter group into one XHTML file ─────────────────
    for chap in tqdm(chapter_groups, desc="Building chapters"):
        chapter_title  = chap["title"]
        html_blocks    = []

        for page in chap["pages"]:
            page_stem     = page["stem"]
            raw_content   = page["content"]
            gemini_photos = page["photos"]
            footnote_text = page["footnotes"]

            # Clean running headers and page-number footers from content
            content_md = _clean_content(raw_content, chapter_title, all_chapter_titles)

            # If the page itself had a sub-section heading, it is now inside
            # content_md as **bold** text — no special treatment needed.
            # But if Gemini still put it in "heading" as a sub-section,
            # prepend it as a sub-heading inside the body instead of a new chapter.
            sub_heading = page["heading"]
            if sub_heading and not _is_true_chapter(sub_heading):
                html_blocks.append(f'<h3>{html.escape(sub_heading)}</h3>')

            # Figure helpers
            page_figs = fig_by_page.get(page_stem, [])

            def replace_photo_placeholder(match):
                idx = int(match.group(1))
                if idx < len(page_figs):
                    fig = page_figs[idx]
                    ep  = _add_figure_to_epub(fig)
                    cap = fig.get("caption") or (gemini_photos[idx].get("caption", "") if idx < len(gemini_photos) else "")
                    cap_html = f"<figcaption>{cap}</figcaption>" if cap else ""
                    return f'\n<figure><img src="{ep}" alt="{fig.get("label", "")}"/>{cap_html}</figure>\n'
                return ""

            parts = re.split(r'(\[PHOTO_\d+\])', content_md)
            for part in parts:
                m = re.match(r'\[PHOTO_(\d+)\]', part)
                if m:
                    html_blocks.append(replace_photo_placeholder(m))
                else:
                    html_blocks.append(_render_markdown_block(part))

            # Append figures that had no inline placeholder
            used = {int(x) for x in re.findall(r'\[PHOTO_(\d+)\]', content_md)}
            for fi, fig in enumerate(page_figs):
                if fi not in used:
                    ep  = _add_figure_to_epub(fig)
                    cap = fig.get("caption") or fig.get("label", "")
                    cap_html = f"<figcaption>{cap}</figcaption>" if cap else ""
                    html_blocks.append(
                        f'\n<figure><img src="{ep}" alt="{fig.get("label", "")}"/>{cap_html}</figure>\n'
                    )

            # Render footnotes in a styled block at the bottom
            if footnote_text and footnote_text.strip():
                html_blocks.append(
                    f'<hr/>\n<p class="footnote">{footnote_text.strip()}</p>'
                )

        html_body = "\n".join(html_blocks).strip()
        if not html_body:
            continue

        full_html = (
            f'<html xmlns="http://www.w3.org/1999/xhtml">\n'
            f'<head><title>{chapter_title}</title>'
            f'<link rel="stylesheet" type="text/css" href="style/main.css"/></head>\n'
            f'<body>\n<h1>{html.escape(chapter_title)}</h1>\n{html_body}\n</body>\n</html>'
        )
        c = epub.EpubHtml(
            title=chapter_title,
            file_name=f"chapter_{chapter_index:03d}.xhtml",
            lang="bn"
        )
        c.content = full_html
        c.add_item(css_item)
        book.add_item(c)
        chapters.append(c)
        chapter_index += 1


    if not chapters:
        print("  No content found to build EPUB.")
        return

    book.toc = tuple(chapters)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    
    spine = ["nav"] + chapters
    if has_cover:
        spine = ["cover"] + spine
    book.spine = spine
    
    epub.write_epub(str(output_file), book, {})
    print(f"  EPUB written: {output_file}  ({len(chapters)} chapters, {len(book_images_added)} images)")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def run(pdf, api_key=None, epub_out=None, progress_callback=None, prefix=None, title=None,
        author="", dpi=150, workers=2, threads=2, sleep=0.0, checkpoint=None):
    """Whole pipeline. api_key=None -> fully offline. checkpoint() runs before each page (pause/stop). Returns the EPUB path."""
    configure(pdf, epub_out, prefix, title, author)
    page_data = pdf_to_images(PDF_PATH, dpi=dpi)
    if api_key:
        run_gemini_ocr(page_data, api_key, workers, progress_callback, checkpoint)
    else:
        print("\n[STAGE 1] Gemini OCR — no API key, running fully offline")
    run_layout_stage(page_data, True, threads, sleep, progress_callback, checkpoint)
    if checkpoint:
        checkpoint()  # a Stop pressed during the last page ends the run here
    figures_index = run_figure_detection(page_data)
    build_epub(page_data, figures_index, OUTPUT_EPUB)
    return OUTPUT_EPUB


def main():
    ap = argparse.ArgumentParser(description="Scanned Bangla PDF -> EPUB (page 1 = cover). Offline on CPU; Gemini optional.")
    ap.add_argument("pdf")
    ap.add_argument("--out", help="output .epub (default: next to the PDF)")
    ap.add_argument("--title", help="book title (default: PDF file name)")
    ap.add_argument("--author", default="")
    ap.add_argument("--prefix", help="cache folder prefix, e.g. 'mahatirtha' reuses mahatirtha_pages/ etc.")
    ap.add_argument("--offline", action="store_true", help="never call Gemini, even if a key exists")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--workers", type=int, default=2, help="parallel Gemini requests")
    ap.add_argument("--threads", type=int, default=2, help="CPU threads for OCR (fewer = cooler, slower)")
    ap.add_argument("--sleep", type=float, default=0.0, help="pause seconds after each OCR'd page")
    args = ap.parse_args()

    if not Path(args.pdf).exists():
        sys.exit(f"ERROR: PDF not found: {args.pdf}")
    out = run(args.pdf, None if args.offline else get_api_key(), args.out, prefix=args.prefix,
              title=args.title, author=args.author, dpi=args.dpi, workers=args.workers,
              threads=args.threads, sleep=args.sleep)
    print(f"\n=== DONE === {out}")


if __name__ == "__main__":
    main()
