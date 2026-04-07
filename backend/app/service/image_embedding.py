import io

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class ImageEmbedding:
    _model = None
    _processor = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            cls._model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        return cls._model

    @classmethod
    def get_processor(cls):
        if cls._processor is None:
            cls._processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return cls._processor

    @classmethod
    def embed(cls, image_bytes: bytes) -> list[float]:
        model = cls.get_model()
        processor = cls.get_processor()

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            features = model.get_image_features(**inputs)

        # 🔥 normalisation (TRÈS IMPORTANT pour cosine similarity)
        features = features / features.norm(p=2, dim=-1, keepdim=True)

        return features[0].tolist()
