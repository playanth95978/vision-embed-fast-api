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
            cls._model = CLIPModel.from_pretrained("models/clip")
        return cls._model

    @classmethod
    def get_processor(cls):
        if cls._processor is None:
            cls._processor = CLIPProcessor.from_pretrained("models/clip")
        return cls._processor

    @classmethod
    def embed(cls, image_bytes: bytes) -> list[float]:
        model = cls.get_model()
        processor = cls.get_processor()

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            features = model.get_image_features(**inputs)

        # 🔒 sécurité : normalisation safe et conversion propre
        features = features.float()
        norm = torch.norm(features, p=2, dim=-1, keepdim=True)
        features = features / (norm + 1e-12)

        # Conversion finale en liste de flottants (1D)
        return features.flatten().detach().cpu().numpy().tolist()
