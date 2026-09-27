# PDF to EPUB AI Converter

A smart desktop application that converts scanned Bengali PDF books into well-formatted EPUB files using the Google Gemini Vision AI model. 

Unlike traditional OCR systems, this tool uses Gemini's advanced spatial reasoning to:
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
- Google Gemini API Key

### Dependencies

Install the required Python packages:
```bash
pip install -r requirements.txt
```
*(If `requirements.txt` is missing, you can manually install: `PyMuPDF Pillow google-genai ebooklib markdown tqdm customtkinter`)*

## Usage

### Using the Desktop App
1. Run the GUI:
   ```bash
   python pdf_to_epub_gui.py
   ```
2. Enter your **Gemini API Key** in the input field.
3. Select your input **PDF** file.
4. Set the name for your output **EPUB**.
5. Click **Start Conversion** and wait for the progress bar to complete.

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
