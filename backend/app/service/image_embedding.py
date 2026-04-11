import io
from typing import List

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class ImageEmbedding:
    _model = None
    _processor = None
    _device = "cuda" if torch.cuda.is_available() else "cpu"

    MODEL_NAME = "openai/clip-vit-base-patch32"

    @classmethod
    def get_model(cls):
        if cls._model is None:
            cls._model = CLIPModel.from_pretrained(cls.MODEL_NAME).to(cls._device)
            cls._model.eval()
        return cls._model

    @classmethod
    def get_processor(cls):
        if cls._processor is None:
            cls._processor = CLIPProcessor.from_pretrained(cls.MODEL_NAME)
        return cls._processor

    @classmethod
    def _normalize(cls, features: torch.Tensor) -> torch.Tensor:
        print("🚀 Initializing an embedding model...")
        print("features =",features)
        features = features.float()
        norm = torch.norm(features, p=2, dim=-1, keepdim=True)
        return features / (norm + 1e-12)

    @classmethod
    def embed_batch(cls, images_bytes: List[bytes]) -> List[List[float]]:
        model = cls.get_model()
        processor = cls.get_processor()

        images = [
            Image.open(io.BytesIO(img)).convert("RGB")
            for img in images_bytes
        ]

        inputs = processor(images=images, return_tensors="pt", padding=True).to(cls._device)

        with torch.no_grad():
            features = model.get_image_features(**inputs)

        features = cls._normalize(features)

        return features.cpu().numpy().tolist()

    @classmethod
    def embed(cls, image_bytes: bytes) -> List[float]:
        return cls.embed_batch([image_bytes])[0]