#!/usr/bin/env python3
"""
pdf_to_epub.py — Convert a PDF (scanned or normal) into an EPUB.

Two output layouts:
  --layout image   Each page becomes a full-page image (fixed layout).
                    Fast, always works, no OCR needed. (default)
  --layout text    OCR every page into real, reflowable, searchable text.
                    Adds: Bangla-aware cleanup, frequency-based spell
                    correction for low-confidence words, an optional
                    ML sentence-plausibility pass, and automatic photo/
                    figure detection + cropping with inline "Fig N"
                    captions placed in reading order.

------------------------------------------------------------------------
Install (core):
    pip install pymupdf ebooklib pillow pytesseract rapidfuzz

Also needs, on the system (not pip):
    tesseract-ocr            # the OCR engine binary
    tesseract-ocr-ben        # Bangla trained data (Debian/Ubuntu package name)
    e.g. sudo apt install tesseract-ocr tesseract-ocr-ben

Install (optional, only for --detect-figures):
    pip install opencv-python-headless

Install (optional, only for --ml-check):
    pip install transformers torch
    (downloads a public Bangla BERT model the first time it runs — needs
     internet access on whatever machine you run this on)
------------------------------------------------------------------------

Usage examples:

  # Fixed-layout, page-image EPUB (no OCR) — works on any PDF
  python3 pdf_to_epub.py book.pdf -o book.epub

  # Reflowable Bangla text EPUB, with spell correction + figure cropping
  python3 pdf_to_epub.py book.pdf -o book.epub --layout text \
      --ocr-lang ben --detect-figures

  # Same, plus a transformer-based sentence plausibility pass
  python3 pdf_to_epub.py book.pdf -o book.epub --layout text \
      --ocr-lang ben --detect-figures --ml-check
"""

import argparse
import io
import os
import re
import sys
import unicodedata
import uuid
from collections import Counter

import fitz  # PyMuPDF
from PIL import Image
from ebooklib import epub


# ---------------------------------------------------------------------------
# Page rendering
# ---------------------------------------------------------------------------

def render_pages(pdf_path, dpi):
    """Yield (page_index, PIL.Image) for every page in the PDF, at `dpi`."""
    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=matrix)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        yield i, img
    doc.close()


def encode_jpeg(img, max_width, quality):
    if max_width and img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Bangla-aware text cleanup
# ---------------------------------------------------------------------------

# Common scanner/OCR noise fixes for Bangla. Extend this table as you find
# more recurring mistakes in your own scans.
_CLEAN_SUBS = [
    (re.compile(r"[\u200b\u200e\u200f]"), ""),          # zero-width / bidi marks
    (re.compile(r"(?<=\S)[\u200c\u200d](?=\s)"), ""),    # stray ZWNJ/ZWJ before space
    (re.compile(r"৷"), "।"),                             # alt danda -> standard danda
    (re.compile(r"[ \t]{2,}"), " "),                      # collapse runs of spaces
    (re.compile(r"\n{3,}"), "\n\n"),                      # collapse blank-line runs
    (re.compile(r"[ \t]+\n"), "\n"),                      # trailing spaces on a line
    (re.compile(r"(?<=[।!?])(?=\S)"), " "),               # ensure space after sentence end
]


def clean_text(text):
    text = unicodedata.normalize("NFC", text)
    for pattern, repl in _CLEAN_SUBS:
        text = pattern.sub(repl, text)
    return text.strip()


_WORD_RE = re.compile(r"[\u0980-\u09FF]+|[A-Za-z]+", re.UNICODE)


def tokenize_words(text):
    return _WORD_RE.findall(text)


# ---------------------------------------------------------------------------
# Frequency-based spell correction
#
# Rather than depending on an external Bangla dictionary (licensing and
# coverage vary a lot), we build a frequency table from the book's own OCR
# output. In any real book the correct spelling of a word vastly outnumbers
# random OCR misreads of it, so rare tokens that are a small edit distance
# from a much more common token are very likely OCR errors of that common
# token. Low-confidence tesseract words are checked first; clearly correct,
# high-confidence words are left alone.
# ---------------------------------------------------------------------------

class SpellCorrector:
    def __init__(self, freq_dict_path=None, min_word_freq=3, max_edit_distance=2):
        self.freq = Counter()
        self.min_word_freq = min_word_freq
        self.max_edit_distance = max_edit_distance
        if freq_dict_path and os.path.isfile(freq_dict_path):
            with open(freq_dict_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2 and parts[-1].isdigit():
                        word = " ".join(parts[:-1])
                        self.freq[word] += int(parts[-1])
                    elif len(parts) == 1:
                        self.freq[parts[0]] += 1

    def learn(self, words):
        self.freq.update(words)

    def correct(self, word):
        """Return the best replacement for `word`, or `word` itself."""
        if len(word) < 2:
            return word
        freq_here = self.freq.get(word, 0)
        if freq_here >= self.min_word_freq:
            return word  # common enough in this document already

        from rapidfuzz import process, fuzz
        candidates = [w for w, c in self.freq.items()
                      if c >= self.min_word_freq and abs(len(w) - len(word)) <= self.max_edit_distance]
        if not candidates:
            return word

        best = process.extractOne(word, candidates, scorer=fuzz.ratio)
        if not best:
            return word
        match, score, _ = best
        # Require a close match and a strictly more common candidate word
        if score >= 80 and match != word and self.freq[match] > max(freq_here, 1):
            return match
        return word


def correct_low_confidence_words(page_words, corrector, conf_threshold):
    """page_words: list of dicts with 'text' and 'conf' from tesseract."""
    out = []
    for w in page_words:
        text, conf = w["text"], w["conf"]
        if text and _WORD_RE.fullmatch(text) and conf < conf_threshold:
            out.append(corrector.correct(text))
        else:
            out.append(text)
    return out


# ---------------------------------------------------------------------------
# Optional ML sentence-plausibility check (transformers, masked LM)
# ---------------------------------------------------------------------------

class MLSentenceChecker:
    """Optional: scores/corrects sentences with a Bangla masked-LM.
    Only activated with --ml-check; safe to skip entirely."""

    def __init__(self, model_name):
        from transformers import pipeline  # imported lazily
        self.fill_mask = pipeline("fill-mask", model=model_name)
        self.mask_token = self.fill_mask.tokenizer.mask_token

    def review_sentence(self, sentence, low_conf_words):
        """For each low-confidence word still present in the sentence, ask
        the model what it thinks belongs there; swap in the model's top
        guess if it's confident and the current word looks wrong."""
        tokens = sentence.split()
        changed = False
        for i, tok in enumerate(tokens):
            if tok not in low_conf_words:
                continue
            masked = tokens.copy()
            masked[i] = self.mask_token
            try:
                preds = self.fill_mask(" ".join(masked))
            except Exception:
                continue
            if preds and preds[0]["score"] > 0.6 and preds[0]["token_str"].strip() != tok:
                tokens[i] = preds[0]["token_str"].strip()
                changed = True
        return " ".join(tokens), changed


# ---------------------------------------------------------------------------
# Figure / photo detection (classic CV — no model download required)
#
# Heuristic: dilate the page's dark pixels so nearby text merges into
# paragraph blobs and photos merge into one solid blob, then classify each
# resulting connected region by how "photographic" it looks: photos have a
# broad, continuous range of gray levels and a high dark-pixel fill ratio,
# while text blocks are sparse, thin strokes on white background.
# ---------------------------------------------------------------------------

def detect_figures(pil_img, min_area_ratio=0.01, max_area_ratio=0.85,
                    fill_ratio_min=0.35, std_min=35):
    import cv2
    import numpy as np

    img = np.array(pil_img.convert("L"))
    h, w = img.shape
    page_area = h * w

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    dilated = cv2.dilate(binary, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        area_ratio = area / page_area
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue
        roi = img[y:y + bh, x:x + bw]
        roi_bin = binary[y:y + bh, x:x + bw]
        fill_ratio = (roi_bin > 0).mean()
        gray_std = roi.std()
        # Photos: dense region (fill_ratio) AND broad continuous-tone
        # variation (gray_std), unlike thin sparse text strokes.
        if fill_ratio >= fill_ratio_min and gray_std >= std_min:
            boxes.append((x, y, bw, bh))

    # Merge boxes that heavily overlap (dilation can split one photo in two)
    boxes = _merge_overlapping(boxes)
    boxes.sort(key=lambda b: b[1])  # reading order, top to bottom
    return boxes


def _merge_overlapping(boxes, iou_thresh=0.15):
    def iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter == 0:
            return 0.0
        union = aw * ah + bw * bh - inter
        return inter / union

    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                if iou(merged[i], merged[j]) > iou_thresh:
                    ax, ay, aw, ah = merged[i]
                    bx, by, bw, bh = merged[j]
                    nx, ny = min(ax, bx), min(ay, by)
                    nx2, ny2 = max(ax + aw, bx + bw), max(ay + ah, by + bh)
                    merged[i] = (nx, ny, nx2 - nx, ny2 - ny)
                    del merged[j]
                    changed = True
                    break
            if changed:
                break
    return merged


# ---------------------------------------------------------------------------
# EPUB assembly — image layout (v1, fixed layout, no OCR)
# ---------------------------------------------------------------------------

def build_image_epub(pdf_path, out_path, title, author, dpi, quality, max_width, make_cover):
    book = _new_book(title, author)
    chapters = []
    cover_set = False

    for idx, img in render_pages(pdf_path, dpi):
        page_no = idx + 1
        print(f"  page {page_no}", end="\r")
        img_bytes = encode_jpeg(img, max_width, quality)
        img_name = f"images/page_{page_no:04d}.jpg"

        if make_cover and not cover_set:
            book.set_cover("cover.jpg", img_bytes)
            cover_set = True

        book.add_item(epub.EpubItem(uid=f"img_{page_no}", file_name=img_name,
                                     media_type="image/jpeg", content=img_bytes))

        chapter = epub.EpubHtml(title=f"Page {page_no}", file_name=f"page_{page_no:04d}.xhtml", lang="und")
        chapter.content = (f'<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Page {page_no}</title>'
                            f'</head><body><div class="page-image"><img src="{img_name}" alt="Page {page_no}"/>'
                            f'</div></body></html>')
        book.add_item(chapter)
        chapters.append(chapter)
    print()

    _finish_book(book, chapters, css_extra="")
    epub.write_epub(out_path, book)
    print(f"Done: {out_path} ({len(chapters)} pages, image layout)")


# ---------------------------------------------------------------------------
# EPUB assembly — text layout (v2: OCR + cleanup + spellcheck + figures)
# ---------------------------------------------------------------------------

def build_text_epub(pdf_path, out_path, title, author, dpi, quality, max_width,
                     ocr_lang, conf_threshold, freq_dict_path, detect_figs,
                     ml_check, ml_model, make_cover, report_path):
    import pytesseract
    from pytesseract import Output

    book = _new_book(title, author)
    chapters = []
    cover_set = False
    corrector = SpellCorrector(freq_dict_path)
    ml_checker = MLSentenceChecker(ml_model) if ml_check else None
    review_lines = []

    print("Pass 1/2: OCR + building frequency table ...")
    pages_data = []
    for idx, img in render_pages(pdf_path, dpi):
        page_no = idx + 1
        print(f"  OCR page {page_no}", end="\r")
        data = pytesseract.image_to_data(img, lang=ocr_lang, output_type=Output.DICT)
        words = [{"text": clean_text(t), "conf": float(c)}
                 for t, c in zip(data["text"], data["conf"]) if t.strip()]
        corrector.learn(w["text"] for w in words if _WORD_RE.fullmatch(w["text"]) and w["conf"] >= conf_threshold)
        figures = detect_figures(img) if detect_figs else []
        pages_data.append({"img": img, "words": words, "figures": figures})
    print()

    print("Pass 2/2: spell-correcting low-confidence words, cropping figures, building EPUB ...")
    fig_counter = 0
    for idx, pdata in enumerate(pages_data):
        page_no = idx + 1
        print(f"  build page {page_no}", end="\r")
        img = pdata["img"]

        corrected_words = correct_low_confidence_words(pdata["words"], corrector, conf_threshold)
        raw_text = " ".join(corrected_words)
        text = clean_text(raw_text)
        sentences = re.split(r"(?<=[।!?])\s+", text)

        if ml_checker and sentences:
            low_conf_set = {w["text"] for w in pdata["words"] if w["conf"] < conf_threshold}
            new_sentences = []
            for s in sentences:
                new_s, changed = ml_checker.review_sentence(s, low_conf_set)
                if changed:
                    review_lines.append(f"[page {page_no}] '{s}' -> '{new_s}'")
                new_sentences.append(new_s)
            sentences = new_sentences

        body_html_parts = [f"<p>{s}</p>" for s in sentences if s.strip()]

        # Crop and embed figures for this page, in reading order.
        fig_html_by_y = []
        for (x, y, w, h) in pdata["figures"]:
            fig_counter += 1
            crop = img.crop((x, y, x + w, y + h))
            fig_bytes = encode_jpeg(crop, max_width, quality)
            fig_name = f"images/fig_{fig_counter:04d}.jpg"
            book.add_item(epub.EpubItem(uid=f"figimg_{fig_counter}", file_name=fig_name,
                                         media_type="image/jpeg", content=fig_bytes))
            fig_html_by_y.append(
                (y, f'<figure><img src="{fig_name}" alt="Fig {fig_counter}"/>'
                    f'<figcaption>Fig {fig_counter}</figcaption></figure>'))

        # Interleave: text first, then figures placed after it in reading
        # order (tesseract paragraph y-positions aren't tracked per-block
        # here, so figures are appended after the page text in the order
        # they were found top-to-bottom; reorder with --detect-figures off
        # if you need figures woven mid-paragraph).
        for _, fig_html in fig_html_by_y:
            body_html_parts.append(fig_html)

        if make_cover and not cover_set and pdata["figures"]:
            x, y, w, h = pdata["figures"][0]
            cover_bytes = encode_jpeg(img.crop((x, y, x + w, y + h)), max_width, quality)
            book.set_cover("cover.jpg", cover_bytes)
            cover_set = True
        elif make_cover and not cover_set:
            book.set_cover("cover.jpg", encode_jpeg(img, max_width, quality))
            cover_set = True

        chapter = epub.EpubHtml(title=f"Page {page_no}", file_name=f"page_{page_no:04d}.xhtml", lang="bn")
        chapter.content = (f'<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Page {page_no}</title>'
                            f'</head><body>{"".join(body_html_parts)}</body></html>')
        book.add_item(chapter)
        chapters.append(chapter)
    print()

    css_extra = """
    body { font-family: serif; line-height: 1.6; }
    p { text-align: justify; margin: 0 0 0.8em 0; }
    figure { text-align: center; margin: 1em 0; }
    figure img { max-width: 100%; height: auto; }
    figcaption { font-size: 0.85em; color: #555; }
    """
    _finish_book(book, chapters, css_extra=css_extra)
    epub.write_epub(out_path, book)
    print(f"Done: {out_path} ({len(chapters)} pages, text layout, {fig_counter} figures)")

    if review_lines and report_path:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(review_lines))
        print(f"ML review notes written to: {report_path}")


# ---------------------------------------------------------------------------
# Shared EPUB helpers
# ---------------------------------------------------------------------------

def _new_book(title, author):
    book = epub.EpubBook()
    book.set_identifier(str(uuid.uuid4()))
    book.set_title(title)
    book.set_language("bn")
    if author:
        book.add_author(author)
    return book


def _finish_book(book, chapters, css_extra=""):
    style = """
    body { margin: 0; padding: 0; }
    .page-image { text-align: center; }
    .page-image img { max-width: 100%; height: auto; display: block; margin: 0 auto; }
    """ + css_extra
    css = epub.EpubItem(uid="style_nav", file_name="style/nav.css", media_type="text/css", content=style)
    book.add_item(css)
    for c in chapters:
        c.add_item(css)
    book.toc = chapters
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav"] + chapters


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Convert a PDF into an EPUB.")
    ap.add_argument("input", help="Path to input PDF")
    ap.add_argument("-o", "--output", help="Path to output EPUB")
    ap.add_argument("--title", help="Book title")
    ap.add_argument("--author", help="Author name")
    ap.add_argument("--dpi", type=int, default=300, help="Render/OCR DPI (default 300)")
    ap.add_argument("--quality", type=int, default=75, help="JPEG quality (default 75)")
    ap.add_argument("--max-width", type=int, default=1400, help="Max embedded image width (0 = no resize)")
    ap.add_argument("--cover", action="store_true", help="Use the first page/figure as EPUB cover")

    ap.add_argument("--layout", choices=["image", "text"], default="image",
                     help="'image': one page-image per page (default, no OCR). "
                          "'text': OCR into reflowable text with cleanup, spellcheck and figures.")

    g = ap.add_argument_group("text-layout options (require --layout text)")
    g.add_argument("--ocr-lang", default="ben", help="Tesseract language code (default 'ben'; use 'ben+eng' for mixed)")
    g.add_argument("--min-confidence", type=float, default=70,
                   help="Tesseract word confidence (0-100) below which a word is treated as 'hard to detect' and spell-corrected (default 70)")
    g.add_argument("--freq-dict", help="Optional path to an external word-frequency file (word<TAB>count per line) to seed spell correction")
    g.add_argument("--detect-figures", action="store_true", help="Auto-detect, crop and embed photos/figures found on each page")
    g.add_argument("--ml-check", action="store_true",
                   help="Extra pass: use a Bangla masked-language-model to sanity-check low-confidence words in context "
                        "(requires `pip install transformers torch`, downloads a model on first run)")
    g.add_argument("--ml-model", default="sagorsarker/bangla-bert-base", help="HuggingFace model id for --ml-check")
    g.add_argument("--review-report", help="Where to write a log of ML-driven corrections (default: <output>.review.txt)")

    args = ap.parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"Input file not found: {args.input}")

    out_path = args.output or (os.path.splitext(args.input)[0] + ".epub")

    title = args.title
    if not title:
        try:
            doc = fitz.open(args.input)
            title = doc.metadata.get("title") or os.path.splitext(os.path.basename(args.input))[0]
            doc.close()
        except Exception:
            title = os.path.splitext(os.path.basename(args.input))[0]

    max_width = args.max_width if args.max_width > 0 else None

    if args.layout == "image":
        build_image_epub(args.input, out_path, title, args.author, args.dpi,
                          args.quality, max_width, args.cover)
    else:
        if args.detect_figures:
            try:
                import cv2  # noqa: F401
            except ImportError:
                sys.exit("Missing dependency: pip install opencv-python-headless")
        if args.ml_check:
            try:
                import transformers  # noqa: F401
            except ImportError:
                sys.exit("Missing dependency: pip install transformers torch")

        review_path = args.review_report or (os.path.splitext(out_path)[0] + ".review.txt")
        build_text_epub(args.input, out_path, title, args.author, args.dpi, args.quality,
                         max_width, args.ocr_lang, args.min_confidence, args.freq_dict,
                         args.detect_figures, args.ml_check, args.ml_model, args.cover, review_path)


if __name__ == "__main__":
    main()
