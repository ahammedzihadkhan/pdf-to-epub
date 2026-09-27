import torch
import torch.nn as nn
from transformers import (
    VisionEncoderDecoderModel,
    AutoTokenizer,
    AutoImageProcessor,
    TrOCRProcessor
)
from PIL import Image

class BanglaTrOCR(nn.Module):
    """
    Next-Gen Transformer-based Bangla OCR Architecture.
    Combines:
    - Vision Encoder: ViT (Vision Transformer) for spatial visual patch extraction
    - Language Decoder: Autoregressive Decoder for sequence generation
    """
    def __init__(self, pretrained_model_name="microsoft/trocr-base-handwritten"):
        super().__init__()
        print(f"Initializing Bangla TrOCR architecture based on {pretrained_model_name}...")
        
        # Explicit component loading to prevent tokenizer wrapper conflicts
        self.image_processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.processor = TrOCRProcessor(image_processor=self.image_processor, tokenizer=self.tokenizer)
        
        self.model = VisionEncoderDecoderModel.from_pretrained(pretrained_model_name)
        
        # Configure generation parameters for Bengali script
        self.model.config.decoder_start_token_id = self.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size
        self.model.config.eos_token_id = self.tokenizer.sep_token_id
        self.model.config.max_length = 128
        self.model.config.early_stopping = True
        self.model.config.no_repeat_ngram_size = 3
        self.model.config.length_penalty = 2.0
        self.model.config.num_beams = 4

    def forward(self, pixel_values, labels=None):
        return self.model(pixel_values=pixel_values, labels=labels)

    def generate(self, pixel_values):
        return self.model.generate(pixel_values)

    def predict_image(self, image: Image.Image) -> str:
        """Run inference on a single cropped text line image."""
        self.eval()
        with torch.no_grad():
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            generated_ids = self.model.generate(pixel_values)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text.strip()

if __name__ == "__main__":
    model = BanglaTrOCR()
    print("BanglaTrOCR model successfully initialized!")
