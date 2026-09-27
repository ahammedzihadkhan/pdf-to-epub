import json
import re
import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import ebooklib
from ebooklib import epub
import markdown

CACHE_DIR = Path("prithibi_ocr_cache")
PAGES_DIR = Path("prithibi_pages")
DATASET_DIR = Path("prithibi_dataset")
PHOTOS_DIR = DATASET_DIR / "photos"
PHOTOS_DIR.mkdir(exist_ok=True)

def extract_photos_from_page(image_path, photo_id_prefix, bbox):
    if not bbox:
        return []
        
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    
    h_img, w_img, _ = img.shape
    
    # Normalize bbox if it's a single list, or handle list of lists
    if isinstance(bbox[0], (int, float)):
        bboxes = [bbox]
    else:
        bboxes = bbox
        
    photo_paths = []
    photo_count = 1
    
    for b in bboxes:
        if len(b) != 4: continue
        x0, y0, x1, y1 = b
        
        # Convert from 0-1000 scale to pixel coordinates
        x_start = int((x0 / 1000.0) * w_img)
        y_start = int((y0 / 1000.0) * h_img)
        x_end = int((x1 / 1000.0) * w_img)
        y_end = int(((y1 + 40) / 1000.0) * h_img) # expand y1 by 4% to include captions
        
        # Clip to image boundaries
        x_start = max(0, x_start)
        y_start = max(0, y_start)
        x_end = min(w_img, x_end)
        y_end = min(h_img, y_end)
        
        if x_end <= x_start or y_end <= y_start:
            continue
            
        photo = img[y_start:y_end, x_start:x_end]
        out_name = f"{photo_id_prefix}_photo_{photo_count}.jpg"
        out_path = PHOTOS_DIR / out_name
        cv2.imwrite(str(out_path), photo)
        photo_paths.append(out_name)
        photo_count += 1
        
    # If no valid crops, copy whole page
    if not photo_paths:
        out_name = f"{photo_id_prefix}_photo_1.jpg"
        out_path = PHOTOS_DIR / out_name
        shutil.copy(image_path, out_path)
        photo_paths.append(out_name)
        
    return photo_paths

def clean_text(text):
    # Fix hyphenated word breaks over newline
    text = re.sub(r'([\u0981-\u09FA]+)[-—][ \t]*\n[ \t]*([\u0981-\u09FA]+)', r'\1\2', text)
    # Fix words split across lines by accident (e.g. "ঐতি \n হ্য" or single newline wraps)
    # This turns single newlines between bengali words into a space
    text = re.sub(r'([\u0981-\u09FA]+)[ \t]*\n[ \t]*([\u0981-\u09FA]+)', r'\1 \2', text)
    # Cleanup any hanging double newlines with spaces
    text = text.replace('\n \n', '\n\n')
    text = text.replace(' \n', '\n')
    return text

def main():
    print("Fixing texts and extracting photos...")
    md_lines = []
    epub_images = []
    
    # Reload from cache
    cache_files = sorted(CACHE_DIR.glob("page_*.json"))
    
    for cache_file in tqdm(cache_files):
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        content = data.get("content_markdown", "")
        if not content:
            continue
            
        # Clean text
        content = clean_text(content)
        
        # Check for photo tags
        # Model might have used random filenames in the tag, so we just check if any photo tag exists
        photo_tags = re.findall(r'!\[(.*?)\]\((.*?)\)', content)
        if photo_tags:
            page_img_path = PAGES_DIR / f"{cache_file.stem}.jpg"
            if page_img_path.exists():
                extracted = extract_photos_from_page(page_img_path, cache_file.stem, data.get("photo_bbox"))
                if extracted:
                    epub_images.extend(extracted)
                    # Replace the markdown tags one by one with the extracted photos
                    # If there are more tags than extracted photos, we reuse the last extracted photo
                    for i, (alt, src) in enumerate(photo_tags):
                        img_to_use = extracted[i] if i < len(extracted) else extracted[-1]
                        # Replace only the first occurrence of this exact src in the content to avoid global replace issues
                        # Actually a simpler replace using the full matched string:
                        original_tag = f"![{alt}]({src})"
                        new_tag = f"![{alt}]({img_to_use})"
                        content = content.replace(original_tag, new_tag, 1)
        
        # Update cache to "remember for future correction"
        data["content_markdown"] = content
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        md_lines.append(content)
        # Instead of `\n---\n` which bothered the user, we just use a page break HTML or two newlines.
        md_lines.append("\n\n")

    # Re-write the full book md
    full_md = "".join(md_lines)
    with open(DATASET_DIR / "full_book.md", "w", encoding="utf-8") as f:
        f.write(full_md)
        
    print("Building EPUB with embedded images...")
    book = epub.EpubBook()
    book.set_identifier('id_prithibi_1234')
    book.set_title('পৃথিবীর পথে পথে')
    book.set_language('bn')
    book.add_author('তারেক অনু')

    # Add images to EPUB
    epub_image_items = {}
    for img_name in set(epub_images):
        img_path = PHOTOS_DIR / img_name
        if img_path.exists():
            with open(img_path, 'rb') as f:
                img_item = epub.EpubItem(uid=img_name, file_name=img_name, media_type='image/jpeg', content=f.read())
                book.add_item(img_item)
                epub_image_items[img_name] = img_item

    chapters = []
    toc = []
    
    # Split by H1 for chapters
    chapters_raw = full_md.split("\n# ")
    
    for idx, chap_raw in enumerate(chapters_raw):
        if not chap_raw.strip():
            continue
        title = chap_raw.split('\n')[0].strip() if idx > 0 else "ভূমিকা"
        chap_md = ("# " if idx > 0 else "") + chap_raw
        
        html_content = markdown.markdown(chap_md, extensions=['tables'])
        c = epub.EpubHtml(title=title, file_name=f'chap_{idx}.xhtml', lang='bn')
        c.content = f'<html><head></head><body>{html_content}</body></html>'
        
        book.add_item(c)
        chapters.append(c)
        toc.append(c)
        
    book.toc = tuple(toc)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ['nav'] + chapters
    epub.write_epub("prithibi_fixed.epub", book, {})
    print("Fixed EPUB generated at prithibi_fixed.epub")

if __name__ == "__main__":
    main()
