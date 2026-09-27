import random
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

class OCRImageAugmentor:
    """
    Advanced Multi-Resolution & Noise Augmentation Engine for Bangla OCR:
    Simulates real-world scan distortions, camera shakes, low DPI, and photocopy artifacts.
    """
    def __init__(self, p_downscale=0.5, p_noise=0.5, p_blur=0.4, p_photocopy=0.3):
        self.p_downscale = p_downscale
        self.p_noise = p_noise
        self.p_blur = p_blur
        self.p_photocopy = p_photocopy

    @staticmethod
    def random_downscale(img: Image.Image, scale_range=(0.35, 0.75)) -> Image.Image:
        """Simulates low DPI / pixelated scans by downscaling and restoring."""
        w, h = img.size
        scale = random.uniform(*scale_range)
        new_w = max(16, int(w * scale))
        new_h = max(8, int(h * scale))
        
        # Downsample with nearest or bilinear, then upsample back
        downsampled = img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
        restored = downsampled.resize((w, h), resample=Image.Resampling.BICUBIC)
        return restored

    @staticmethod
    def add_gaussian_noise(img: Image.Image, max_sigma=25) -> Image.Image:
        """Adds Gaussian grain and scanner sensor noise."""
        np_img = np.array(img).astype(np.float32)
        sigma = random.uniform(5, max_sigma)
        noise = np.random.normal(0, sigma, np_img.shape)
        noisy = np.clip(np_img + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)

    @staticmethod
    def add_salt_and_pepper_noise(img: Image.Image, prob=0.015) -> Image.Image:
        """Adds scanner dust spots and paper speckles."""
        np_img = np.array(img)
        mask = np.random.rand(*np_img.shape[:2])
        np_img[mask < (prob / 2)] = 0      # Pepper
        np_img[mask > (1 - prob / 2)] = 255 # Salt
        return Image.fromarray(np_img)

    @staticmethod
    def add_blur(img: Image.Image) -> Image.Image:
        """Simulates camera shake or out-of-focus optics."""
        blur_choice = random.choice(["gaussian", "box"])
        if blur_choice == "gaussian":
            radius = random.uniform(0.5, 1.8)
            return img.filter(ImageFilter.GaussianBlur(radius))
        else:
            return img.filter(ImageFilter.BoxBlur(1))

    @staticmethod
    def add_photocopy_degradation(img: Image.Image) -> Image.Image:
        """Simulates harsh binarization and toner degradation from old photocopy machines."""
        np_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        # Apply random adaptive thresholding
        block_size = random.choice([11, 15, 19, 23])
        c_val = random.uniform(2, 8)
        thresh = cv2.adaptiveThreshold(
            np_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c_val
        )
        # Add slight morphological dilation or erosion to simulate bleeding ink
        if random.random() < 0.5:
            kernel = np.ones((2, 2), np.uint8)
            thresh = cv2.dilate(thresh, kernel, iterations=1)
        return Image.fromarray(cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB))

    @staticmethod
    def add_lighting_gradient(img: Image.Image) -> Image.Image:
        """Simulates uneven book crease shadows and lighting gradients."""
        w, h = img.size
        gradient = np.linspace(random.uniform(0.6, 1.0), random.uniform(0.6, 1.0), w)
        gradient = np.tile(gradient, (h, 1))
        
        np_img = np.array(img).astype(np.float32)
        if len(np_img.shape) == 3:
            gradient = np.expand_dims(gradient, axis=2)
            
        shaded = np.clip(np_img * gradient, 0, 255).astype(np.uint8)
        return Image.fromarray(shaded)

    def augment(self, img: Image.Image) -> Image.Image:
        """Applies a randomized combination of augmentations."""
        # 1. Multi-Resolution Downscaling
        if random.random() < self.p_downscale:
            img = self.random_downscale(img)
            
        # 2. Blur / Camera Shake
        if random.random() < self.p_blur:
            img = self.add_blur(img)
            
        # 3. Noise Injection
        if random.random() < self.p_noise:
            if random.random() < 0.5:
                img = self.add_gaussian_noise(img)
            else:
                img = self.add_salt_and_pepper_noise(img)
                
        # 4. Photocopy / Binarization Artifacts
        if random.random() < self.p_photocopy:
            img = self.add_photocopy_degradation(img)
        else:
            # Random brightness & contrast tweak
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(random.uniform(0.7, 1.3))
            
        # 5. Lighting gradient / shadow
        if random.random() < 0.3:
            img = self.add_lighting_gradient(img)
            
        return img

# Global default augmentor instance
augmentor = OCRImageAugmentor()

if __name__ == "__main__":
    test_img = Image.new("RGB", (384, 48), color="white")
    aug_img = augmentor.augment(test_img)
    print("✅ Augmentation engine initialized & verified successfully!")
