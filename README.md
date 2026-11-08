# PDF to EPUB AI Converter

A smart desktop application that converts scanned Bengali PDF books into well-formatted EPUB files using the Google Gemini Vision AI model. 

## Quick start: offline, low-resource (CPU only, 8 GB RAM)

```bash
python -m bangla_ai_ocr.layout_model train          # once, a few minutes on 2 CPU threads
python -m bangla_ai_ocr.text_corrector collect      # optional, slow (~1-2 min/page); resumable
python -m bangla_ai_ocr.text_corrector train        # learns offline text fixes from Gemini pages
python bangla_pdf_to_epub.py "book.pdf" --offline --title "বইয়ের নাম" --author "লেখক" --sleep 0.5
```

- Page 1 always becomes the EPUB cover.
- A small local layout model finds photos. EasyOCR reads the text with photos blanked out, so no internet is needed.
- A learned text corrector fixes words EasyOCR often misreads. It is only switched on if it lowers the error rate on held-out pages.
- With a Gemini key, Gemini reads the text and the local model is re-checked against it.
- Use `--threads 1` and a larger `--sleep` to slow down further on a hot laptop.
- Reuse old caches with `--prefix mahatirtha`.
- Design and ER diagram: [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md).

With a Gemini key, the tool also uses Gemini's spatial reasoning. Unlike traditional OCR systems, this tool uses Gemini's advanced spatial reasoning to:
- Extract text with exceptionally high accuracy.
- Fix Bengali spelling mistakes.
- Automatically detect chapters based on text headings.
- Intelligently crop illustrations/photos and place them in the correct position inside the EPUB.
- Generate high-quality machine learning datasets (`ocr_dataset.jsonl` and `spell_correction_dataset.jsonl`) from the OCR results for future fine-tuning.

## Features

- **Modern GUI:** Built with `customtkinter` for a beautiful, responsive desktop experience.
- **AI-Powered OCR:** Uses `gemini-3.5-flash` for fast and accurate Bengali transcription.
- **Smart Photo Extraction:** Crops images directly from the PDF and anchors them within the EPUB content flow.
- **Chapter Detection:** Automatically splits pages into chapters by recognizing large chapter headings.
- **Caching Mechanism:** Saves progress locally (`prithibi_ocr_cache/`) so that if the process is interrupted, it resumes exactly where it left off.
- **Concurrency:** Processes multiple pages simultaneously to significantly reduce conversion time.

## Requirements

- Python 3.10+
- Optional: Google Gemini API Key (better text; leave empty to run offline)

### Dependencies

Install the required Python packages:
```bash
pip install -r requirements.txt
```
*(If `requirements.txt` is missing, you can manually install: `PyMuPDF Pillow google-genai ebooklib markdown tqdm customtkinter opencv-python scikit-learn easyocr`)*

## Usage

### Using the Desktop App
1. Run the GUI:
   ```bash
   python pdf_to_epub_gui.py
   ```
2. Click **Browse** and choose the scanned PDF. Page 1 is previewed as the cover, and the EPUB path and title are filled in.
3. Choose the **Text reader**. **Offline** (the default) needs no internet. **Gemini** sends pages to Gemini and needs an API key.
4. Set **CPU threads** and **Rest per page** to suit your PC. Fewer threads and longer rests keep a laptop cool.
5. Click **Start conversion**. The **Preview** tab follows the page being read and draws red boxes around the photos found. Flip pages with the slider or the arrow keys, or click a page to open it full size. The progress card shows the current step, pages done, elapsed time, time left and speed. The **Log** tab shows each step.
   **Pause** holds the run after the current page (**Resume** continues). **Stop** ends it after the current page; finished pages are kept, and **Resume conversion** picks up from there.
6. When it finishes, click **Open EPUB** or **Open folder**. You can also close the app at any time, and the next run resumes.

### Creating an Executable (.exe)
You can build a standalone Windows application that doesn't require Python to be installed.
1. Install PyInstaller:
   ```bash
   pip install pyinstaller
   ```
2. Build the app:
   ```bash
   pyinstaller --noconfirm --onedir --windowed --name "PDF_to_EPUB" pdf_to_epub_gui.py
   ```
3. The executable will be available inside the `dist/PDF_to_EPUB/` folder.

## Dataset Generation
As a byproduct of the conversion process, two machine-learning datasets are generated in the `prithibi_dataset/` directory:
- `ocr_dataset.jsonl`: Contains the raw text and the corresponding image filename.
- `spell_correction_dataset.jsonl`: Maps raw OCR text with spelling errors to the grammatically corrected text.

## Limitations
- Large PDFs (300+ pages) will require a stable internet connection and take some time (around 10-15 minutes) depending on your API rate limits.
- Currently optimized for Bengali language extraction.

## License
MIT License
