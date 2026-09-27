import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os

PAGES_DIR = Path("pages")
OUTPUT_DIR = Path("images")
OUTPUT_DIR.mkdir(exist_ok=True)

def extract_photos_from_page(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Simple thresholding
    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological closing to fuse text lines vs photos
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    h_img, w_img = gray.shape
    total_area = h_img * w_img
    
    photo_paths = []
    photo_count = 1
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        
        # Look for large areas that might be photos (e.g. > 15% of page)
        if area > total_area * 0.15 and area < total_area * 0.95:
            # Check edge density or intensity variance to avoid false positives (like large blank boxes)
            roi_gray = gray[y:y+h, x:x+w]
            edges = cv2.Canny(roi_gray, 100, 200)
            edge_density = np.sum(edges > 0) / area
            
            # Photos usually have a reasonable edge density, text blocks have high edge density
            # Let's just save large contiguous blocks that are solid
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0: continue
            solidity = float(area) / hull_area
            
            if solidity > 0.85:
                # To distinguish from a block of text, we can check standard deviation
                std_dev = np.std(roi_gray)
                if std_dev > 20: # Photos have variance, solid black boxes have 0
                    photo = img[y:y+h, x:x+w]
                    out_name = f"{image_path.stem}_photo_{photo_count}.jpg"
                    out_path = OUTPUT_DIR / out_name
                    cv2.imwrite(str(out_path), photo)
                    photo_paths.append(out_path)
                    photo_count += 1
                
    return photo_paths

def main():
    image_files = sorted(PAGES_DIR.glob("page_*.jpg"))
    print(f"Scanning {len(image_files)} pages for photos...")
    
    total_photos = 0
    for img_path in tqdm(image_files):
        paths = extract_photos_from_page(img_path)
        if paths:
            total_photos += len(paths)
            
    print(f"Extracted {total_photos} photos total.")

if __name__ == "__main__":
    main()
