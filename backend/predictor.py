"""
Inference and Prediction Pipeline for BloomAI Flower Classification.
Handles checkpoint downloading/loading, image preprocessing, and inference.
"""
import io
import json
import os
import urllib.request
from typing import Dict, Any, List, Tuple
from PIL import Image, UnidentifiedImageError

import torch
from torchvision import transforms

from model import create_flower_resnet18

# Configuration paths & environment variables
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
CONFIG_PATH = os.path.join(PROJECT_DIR, "model", "model_config.json")

# Candidate local checkpoint paths (searches local model folder and backup folder)
DEFAULT_LOCAL_PATHS = [
    os.path.join(PROJECT_DIR, "model", "best_resnet_model.pth"),
    os.path.join(BASE_DIR, "best_resnet_model.pth"),
    os.path.join(PROJECT_DIR, "..", "flower_classification_results", "best_resnet_model.pth"),
    os.path.join(PROJECT_DIR, "..", "Flower_Project_Backup", "flower_classification_results", "best_resnet_model.pth"),
]

# Supported MIME / Image extensions
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit


class FlowerPredictor:
    def __init__(self, config_path: str = CONFIG_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = self._load_config(config_path)
        self.classes: List[str] = self.config.get("classes", ["daisy", "dandelion", "rose", "sunflower", "tulip"])
        self.model = None
        self.is_loaded = False
        self.load_error = None
        self.checkpoint_source = None

        # Build verified torchvision transform:
        # Resize to 256, CenterCrop to 224, ToTensor, Normalize with ImageNet mean & std
        mean = self.config.get("image_mean", [0.485, 0.456, 0.406])
        std = self.config.get("image_std", [0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        # Attempt initial model loading
        self._initialize_model()

    def _load_config(self, path: str) -> Dict[str, Any]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "model_name": "ResNet-18",
            "task": "Five-Class Flower Classification",
            "classes": ["daisy", "dandelion", "rose", "sunflower", "tulip"],
            "input_size": [3, 224, 224],
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
        }

    def _find_checkpoint(self) -> str:
        # 1. Environment variable MODEL_CHECKPOINT_PATH
        env_path = os.environ.get("MODEL_CHECKPOINT_PATH")
        if env_path and os.path.exists(env_path):
            return env_path

        # 2. Check candidate default local paths
        for p in DEFAULT_LOCAL_PATHS:
            normalized = os.path.normpath(p)
            if os.path.exists(normalized):
                return normalized

        # 3. Download from HF URL if configured
        hf_url = os.environ.get("MODEL_DOWNLOAD_URL")
        if hf_url:
            target_path = os.path.join(BASE_DIR, "downloaded_resnet.pth")
            if not os.path.exists(target_path):
                print(f"Downloading checkpoint from {hf_url}...")
                req = urllib.request.Request(hf_url)
                hf_token = os.environ.get("HF_TOKEN")
                if hf_token:
                    req.add_header("Authorization", f"Bearer {hf_token}")
                with urllib.request.urlopen(req) as response, open(target_path, "wb") as out_file:
                    out_file.write(response.read())
            return target_path

        return None

    def _initialize_model(self):
        try:
            ckpt_path = self._find_checkpoint()
            if not ckpt_path:
                self.load_error = "Checkpoint not found in environment or local default paths."
                self.is_loaded = False
                return

            self.model = create_flower_resnet18(num_classes=len(self.classes))
            state_dict = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=True)
            self.model.to(self.device)
            self.model.eval()

            self.is_loaded = True
            self.checkpoint_source = os.path.basename(ckpt_path)
            self.load_error = None
            print(f"Successfully loaded ResNet-18 checkpoint from: {ckpt_path}")
        except Exception as e:
            self.is_loaded = False
            self.load_error = str(e)
            print(f"Error loading checkpoint: {e}")

    def predict_image_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError(f"Model is not loaded: {self.load_error or 'Unknown error'}")

        if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
            raise ValueError("File size exceeds 10 MB limit.")

        try:
            image = Image.open(io.BytesIO(image_bytes))
            # Convert to RGB to handle grayscale (1 ch) or RGBA (4 ch)
            if image.mode != "RGB":
                image = image.convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            raise ValueError("Invalid or corrupted image format. Please upload JPG, PNG, or WEBP.")

        # Preprocess
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1).squeeze(0).cpu().tolist()

        # Build results
        all_probabilities = {
            self.classes[i]: round(float(prob), 4)
            for i, prob in enumerate(probabilities)
        }

        # Sort classes by probability descending
        sorted_pairs = sorted(
            [(self.classes[i], float(prob)) for i, prob in enumerate(probabilities)],
            key=lambda item: item[1],
            reverse=True
        )

        top_pred_class, top_pred_prob = sorted_pairs[0]

        top_three = [
            {
                "class_name": c,
                "confidence": round(p * 100, 2),
                "probability": round(p, 4)
            }
            for c, p in sorted_pairs[:3]
        ]

        return {
            "predicted_class": top_pred_class,
            "confidence_percentage": round(top_pred_prob * 100, 2),
            "top_three": top_three,
            "all_probabilities": all_probabilities
        }

