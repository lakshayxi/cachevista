"""
CacheVista Encoder Module
Loads CLIP ViT-L/14 and converts images to 768-dim embeddings.
"""

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPImageProcessor


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class CLIPEncoder:
    def __init__(self):
        print("Loading CLIP ViT-L/14 ...")
        self.device = get_device()
        print(f"Using device: {self.device}")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.model = self.model.to(self.device)
        self.model.eval()
        print("CLIP loaded!")

    def encode(self, image: Image.Image) -> np.ndarray:
        # CLIPImageProcessor handles images only - no text padding issues
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().cpu().numpy().astype(np.float32)
