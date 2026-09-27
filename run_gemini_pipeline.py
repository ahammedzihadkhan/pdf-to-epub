import os
import sys
import json
import time
from pathlib import Path
import concurrent.futures

try:
    if sys.stdout is not None:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import fitz  # PyMuPDF
from PIL import Image
from google import genai
from google.genai import types
from tqdm import tqdm

import ebooklib
from ebooklib import epub
import markdown

def get_api_key():
    key = os.environ.get("GEMINI_API_KEY")
    if key: return key
    if Path(".env").exists():
        for line in open(".env"):
            if line.startswith("GEMINI_API_KEY="):
                return line.strip().split("=")[1]
    return "YOUR_API_KEY"

API_KEY = get_api_key()

# Default Configuration
DEFAULT_PDF_PATH = Path("পৃথিবীর পথে পথে - তারেক অনু.pdf")
DEFAULT_OUTPUT = "prithibi.epub"
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

OCR_PROMPT = """You are an expert transcriber and EPUB formatter for Bengali books.
Transcribe this scanned page from 'পৃথিবীর পথে পথে' by তারেক অনু.

RULES:
1. Transcribe ALL Bengali and English text verbatim. Fix minor OCR mistakes but keep the tone intact.
2. **CHAPTERS**: If this page has a large heading that looks like the start of a new chapter or article, put it in the "heading" field. Otherwise, set it to null.
3. **PHOTOS**: If the page has distinct photographs or illustrations:
   - For each photo, you MUST insert a placeholder like `[PHOTO_0]`, `[PHOTO_1]` etc. inside `content_markdown` EXACTLY where it appears relative to the text flow.
   - Provide the bounding box for each photo in the `photos` list in the format [ymin, xmin, ymax, xmax] using normalized coordinates (0 to 1000).

4. Return ONLY valid JSON:
{
  "heading": "string or null",
  "raw_text": "Raw text before spell checking",
  "content_markdown": "Full transcription WITH spelling corrected. Must contain [PHOTO_0] etc. if photos exist.",
  "photos": [
    {
      "box_2d": [ymin, xmin, ymax, xmax]
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
    book.set_identifier('id_book_1234')
    book.set_title(Path(output_file).stem)
    book.set_language('bn')
    book.add_author('Unknown')

    chapters = []
    current_chapter = None
    current_chapter_title = "সূচনা (Introduction)"
    current_chapter_md = ""
    chapter_index = 1
    
    def finalize_chapter():
        nonlocal current_chapter_md, current_chapter_title, chapter_index, chapters, book
        if not current_chapter_md.strip():
            return
        html_content = markdown.markdown(current_chapter_md, extensions=['tables'])
        c = epub.EpubHtml(title=current_chapter_title, file_name=f"chapter_{chapter_index:03d}.xhtml", lang='bn')
        c.content = f'<html><head><title>{current_chapter_title}</title></head><body><h1>{current_chapter_title}</h1>{html_content}</body></html>'
        book.add_item(c)
        chapters.append(c)
        current_chapter_md = ""
        chapter_index += 1

    for p in page_data:
        cache_file = OCR_CACHE / f"{p['img_path'].stem}.json"
        if not cache_file.exists():
            continue
            
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
            
        heading = data.get("heading")
        content_md = data.get("content_markdown", "")
        photos = data.get("photos", [])
        
        if heading and heading.strip() and heading.lower() != "null":
            finalize_chapter()
            current_chapter_title = heading.strip()
            
        # Crop photos and replace placeholders
        img_w, img_h = 0, 0
        pil_img = None
        if photos:
            try:
                pil_img = Image.open(p['img_path'])
                img_w, img_h = pil_img.size
            except:
                pass
            
        for f_idx, photo in enumerate(photos):
            box = photo.get("box_2d")
            photo_filename = f"{p['img_path'].stem}_fig_{f_idx}.jpg"
            photo_path = PHOTOS_DIR / photo_filename
            
            valid_crop = False
            if box and len(box) == 4 and pil_img:
                ymin, xmin, ymax, xmax = box
                y0 = max(0, int(ymin * img_h / 1000))
                x0 = max(0, int(xmin * img_w / 1000))
                y1 = min(img_h, int(ymax * img_h / 1000))
                x1 = min(img_w, int(xmax * img_w / 1000))
                
                if x1 > x0 and y1 > y0:
                    crop = pil_img.crop((x0, y0, x1, y1))
                    crop.save(photo_path, format="JPEG", quality=85)
                    valid_crop = True
            
            if valid_crop:
                with open(photo_path, "rb") as pf:
                    book.add_item(epub.EpubItem(uid=photo_filename, file_name=f"images/{photo_filename}", media_type="image/jpeg", content=pf.read()))
                
                img_tag = f'\n\n<figure><img src="images/{photo_filename}" style="max-width:100%; height:auto;" /></figure>\n\n'
                
                placeholder = f"[PHOTO_{f_idx}]"
                if placeholder in content_md:
                    content_md = content_md.replace(placeholder, img_tag)
                else:
                    # If model forgot placeholder, append it
                    content_md += img_tag

        current_chapter_md += f"\n\n{content_md}\n\n"

    # Finalize the last chapter
    finalize_chapter()

    if chapters:
        book.toc = tuple(chapters)
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        book.spine = ['nav'] + chapters
        epub.write_epub(output_file, book, {})
        print(f"EPUB generated at {output_file}")
    else:
        print("No content found to build EPUB.")

def run_pipeline(pdf_path: str, api_key: str, output_epub: str, progress_callback=None):
    pdf_path_obj = Path(pdf_path)
    if not pdf_path_obj.exists():
        print(f"File not found: {pdf_path_obj}")
        return

    page_data = pdf_to_images(pdf_path_obj)
    client = genai.Client(api_key=api_key)
    
    to_process = [p for p in page_data if not (OCR_CACHE / f"{p['img_path'].stem}.json").exists()]
    
    if to_process:
        print(f"Processing {len(to_process)} pages with Gemini OCR concurrently...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(ocr_single_page, client, p['img_path']): p for p in to_process}
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(to_process))
            
    process_photos_and_build_epub(page_data, output_file=output_epub)
    print("Pipeline completed successfully!")

def main():
    run_pipeline(str(DEFAULT_PDF_PATH), API_KEY, DEFAULT_OUTPUT)

if __name__ == "__main__":
    main()
