import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import io
import concurrent.futures

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import fitz  # PyMuPDF
from PIL import Image
from google import genai
from google.genai import types
from tqdm import tqdm

import ebooklib
from ebooklib import epub
import markdown

API_KEY = os.environ.get("GEMINI_API_KEY") or "YOUR_API_KEY"

# Configuration
PDF_PATH = Path("পৃথিবীর পথে পথে - তারেক অনু.pdf")
PAGES_DIR = Path("prithibi_pages")
PHOTOS_DIR = Path("prithibi_photos")
OCR_CACHE = Path("prithibi_ocr_cache")
DATASET_DIR = Path("prithibi_dataset")

PAGES_DIR.mkdir(exist_ok=True)
PHOTOS_DIR.mkdir(exist_ok=True)
OCR_CACHE.mkdir(exist_ok=True)
DATASET_DIR.mkdir(exist_ok=True)

OCR_DATASET_FILE = DATASET_DIR / "ocr_dataset.jsonl"
SPELL_DATASET_FILE = DATASET_DIR / "spell_correction_dataset.jsonl"

MODELS = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.7-flash",
]

OCR_PROMPT = """You are an expert transcriber and spell-checker for Bengali books.
Transcribe this scanned page from the travelogue 'পৃথিবীর পথে পথে' by তারেক অনু.

RULES:
1. Transcribe ALL Bengali and English text verbatim.
2. Fix minor OCR/spelling mistakes in the Bengali text (spell-check), but keep the sentence structure and author's tone intact.
3. If the page contains a distinct photograph or illustration (not just text), describe it and provide its bounding box.
   The bounding box must be in the format [ymin, xmin, ymax, xmax] using normalized coordinates from 0 to 1000 (e.g. [150, 100, 450, 900]).
   Do NOT transcribe text that is part of the photograph.
4. Return ONLY a valid JSON object in this schema:
{
  "heading": "string or null",
  "page_number": "string (original printed page number if visible, or null)",
  "raw_text": "The raw extracted text BEFORE spell checking (for dataset generation)",
  "content_markdown": "Full transcription in markdown WITH spelling corrected",
  "photos": [
    {
      "description": "Short description of the photo",
      "box_2d": [150, 100, 450, 900]
    }
  ]
}"""

def pdf_to_images(pdf_path: Path):
    print(f"Extracting pages from {pdf_path.name}...")
    doc = fitz.open(pdf_path)
    page_data = []
    
    for i in tqdm(range(len(doc)), desc="PDF -> Images"):
        page = doc[i]
        pix = page.get_pixmap(dpi=150)
        img_path = PAGES_DIR / f"page_{i:04d}.jpg"
        if not img_path.exists():
            pix.save(str(img_path))
        page_data.append({
            "img_path": img_path,
            "page_num": i
        })
    return page_data

def ocr_single_page(client: genai.Client, image_path: Path):
    cache_file = OCR_CACHE / f"{image_path.stem}.json"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    for model_name in MODELS:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                        OCR_PROMPT,
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1,
                    )
                )
                text = response.text.strip()
                data = json.loads(text)
                
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                    
                if data.get("raw_text") and data.get("content_markdown"):
                    with open(OCR_DATASET_FILE, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"image": image_path.name, "text": data["raw_text"]}, ensure_ascii=False) + "\n")
                    with open(SPELL_DATASET_FILE, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"bad_text": data["raw_text"], "corrected_text": data["content_markdown"]}, ensure_ascii=False) + "\n")
                return data
            except Exception as e:
                time.sleep(2.0)
    return None

def process_photos_and_build_epub(page_data, output_file="prithibi.epub"):
    print("Cropping photos and building EPUB...")
    book = epub.EpubBook()
    book.set_identifier('id_prithibi_1234')
    book.set_title('পৃথিবীর পথে পথে')
    book.set_language('bn')
    book.add_author('তারেক অনু')

    chapters = []
    
    for p in page_data:
        cache_file = OCR_CACHE / f"{p['img_path'].stem}.json"
        content_md = ""
        photos = []
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    content_md = data.get("content_markdown", "")
                    photos = data.get("photos", [])
            except Exception:
                pass
        
        # Crop photos based on box_2d
        img_w, img_h = 0, 0
        pil_img = None
        if photos:
            pil_img = Image.open(p['img_path'])
            img_w, img_h = pil_img.size
            
        photo_files = []
        for f_idx, photo in enumerate(photos):
            box = photo.get("box_2d")
            if not box or len(box) != 4: continue
            
            ymin, xmin, ymax, xmax = box
            # convert normalized (0-1000) to pixels
            y0 = max(0, int(ymin * img_h / 1000))
            x0 = max(0, int(xmin * img_w / 1000))
            y1 = min(img_h, int(ymax * img_h / 1000))
            x1 = min(img_w, int(xmax * img_w / 1000))
            
            if x1 <= x0 or y1 <= y0: continue
            
            crop = pil_img.crop((x0, y0, x1, y1))
            photo_filename = f"{p['img_path'].stem}_fig_{f_idx}.jpg"
            photo_path = PHOTOS_DIR / photo_filename
            crop.save(photo_path, format="JPEG", quality=85)
            photo_files.append(photo_filename)
        
        # Embed photos in EPUB
        for photo in photo_files:
            with open(PHOTOS_DIR / photo, "rb") as pf:
                book.add_item(epub.EpubItem(uid=photo, file_name=f"images/{photo}", media_type="image/jpeg", content=pf.read()))
            content_md += f"\n\n<figure><img src=\"images/{photo}\" /></figure>\n\n"
            
        if not content_md.strip() and not photo_files:
            continue
            
        html_content = markdown.markdown(content_md, extensions=['tables'])
        c = epub.EpubHtml(title=f"Page {p['page_num']}", file_name=f"page_{p['page_num']:04d}.xhtml", lang='bn')
        c.content = f'<html><head></head><body>{html_content}</body></html>'
        book.add_item(c)
        chapters.append(c)

    book.toc = tuple(chapters)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ['nav'] + chapters
    epub.write_epub(output_file, book, {})
    print(f"EPUB generated at {output_file}")

def main():
    if not PDF_PATH.exists():
        print(f"File not found: {PDF_PATH}")
        sys.exit(1)

    page_data = pdf_to_images(PDF_PATH)
    client = genai.Client(api_key=API_KEY)
    
    to_process = [p for p in page_data if not (OCR_CACHE / f"{p['img_path'].stem}.json").exists()]
    
    if to_process:
        print(f"Processing {len(to_process)} pages with Gemini OCR concurrently...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(ocr_single_page, client, p['img_path']): p for p in to_process}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="OCR Progress"):
                pass
            
    process_photos_and_build_epub(page_data)

if __name__ == "__main__":
    main()
