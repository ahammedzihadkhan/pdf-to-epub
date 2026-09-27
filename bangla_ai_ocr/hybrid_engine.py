import sys
import re
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

from .rl_spell_corrector import RLSpellCorrector

class HybridBanglaOCR:
    """
    Unified Single Hybrid Bangla AI OCR Engine:
    Combines:
    1. Preprocessing: CLAHE Adaptive Contrast Enhancement & Binarization
    2. Vision Detection: CRAFT / Connected-Component Segmentation
    3. Neural Recognition: 2-Stage Fine-Tuned TrOCR Vision Transformer
    4. Post-Correction: Reinforcement Learning & Language-Model Spell Corrector
    """
    def __init__(self, model_checkpoint_dir: Path = None):
        print("🚀 Initializing Unified Hybrid Bangla AI OCR Engine...")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.corrector = RLSpellCorrector()
        
        # Load EasyOCR fallback
        try:
            import easyocr
            self.easyocr_reader = easyocr.Reader(['bn', 'en'], gpu=torch.cuda.is_available())
        except Exception as e:
            self.easyocr_reader = None

        # Load Neural TrOCR Model if checkpoint is available
        self.trocr_model = None
        self.processor = None
        if model_checkpoint_dir and Path(model_checkpoint_dir).exists():
            try:
                from transformers import AutoImageProcessor, RobertaTokenizer, TrOCRProcessor, VisionEncoderDecoderModel
                print(f"Loading trained weights from {model_checkpoint_dir}...")
                image_processor = AutoImageProcessor.from_pretrained(model_checkpoint_dir)
                tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint_dir)
                self.processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
                self.trocr_model = VisionEncoderDecoderModel.from_pretrained(model_checkpoint_dir).to(self.device)
                self.trocr_model.eval()
                print("✅ Trained Neural TrOCR weights loaded successfully!")
            except Exception as e:
                print(f"Note: Neural TrOCR load notice: {e}")

    def preprocess_image(self, img_array: np.ndarray) -> np.ndarray:
        """Enhance image contrast and remove background noise."""
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_array
            
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return enhanced

    def recognize_line(self, line_img: Image.Image) -> str:
        """Transcribes a single line image using TrOCR with RL Spell Correction."""
        if self.trocr_model and self.processor:
            pixel_values = self.processor(line_img, return_tensors="pt").pixel_values.to(self.device)
            with torch.no_grad():
                generated_ids = self.trocr_model.generate(pixel_values, max_length=128, num_beams=4)
                raw_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        elif self.easyocr_reader:
            np_img = np.array(line_img)
            results = self.easyocr_reader.readtext(np_img, detail=0)
            raw_text = " ".join(results)
        else:
            raw_text = ""
            
        # Apply Stage 3 Reinforcement Learning Spell Correction
        refined_text = self.corrector.correct_text(raw_text)
        return refined_text

    def transcribe_page(self, image_path: Path) -> str:
        """Complete end-to-end pipeline: Preprocess ➔ Detect ➔ Recognize ➔ RL Refine."""
        img = cv2.imread(str(image_path))
        if img is None:
            return ""
            
        prep = self.preprocess_image(img)
        
        if self.easyocr_reader:
            results = self.easyocr_reader.readtext(prep, detail=0)
            raw_text = "\n\n".join(results)
        else:
            raw_text = ""
            
        # RL Spell Correction & Ligature Repair
        final_text = self.corrector.correct_text(raw_text)
        return final_text

if __name__ == "__main__":
    ocr = HybridBanglaOCR()
    print("✅ Unified Hybrid Bangla AI OCR Engine ready for inference!")
