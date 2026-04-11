import io
from typing import List

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class ImageEmbedding:
    """
    Classe utilitaire pour la gestion des embeddings d'images utilisant le modèle CLIP d'OpenAI.
    Le modèle CLIP (Contrastive Language-Image Pre-training) permet de projeter des images
    dans un espace vectoriel sémantique.
    """
    _model = None
    _processor = None
    # Utilise le GPU (cuda) si disponible, sinon bascule sur le CPU
    _device = "cuda" if torch.cuda.is_available() else "cpu"

    # Modèle CLIP de base utilisant un ViT-B/32 (Vision Transformer)
    MODEL_NAME = "openai/clip-vit-base-patch32"

    @classmethod
    def get_model(cls):
        """
        Initialise et retourne l'instance unique (Singleton) du modèle CLIP.
        """
        if cls._model is None:
            # Chargement du modèle pré-entraîné et transfert vers le périphérique (CPU/GPU)
            cls._model = CLIPModel.from_pretrained(cls.MODEL_NAME).to(cls._device)
            # Passage en mode évaluation (désactive le dropout, etc.)
            cls._model.eval()
        return cls._model

    @classmethod
    def get_processor(cls):
        """
        Initialise et retourne l'instance unique du processeur CLIP (gère le redimensionnement, la normalisation, etc.)
        """
        if cls._processor is None:
            cls._processor = CLIPProcessor.from_pretrained(cls.MODEL_NAME)
        return cls._processor

    @classmethod
    def _normalize(cls, features: torch.Tensor) -> torch.Tensor:
        """
        Normalise les vecteurs de caractéristiques (L2 norm) pour permettre la recherche par similarité cosinus.
        """
        print("🚀 Initializing an embedding model...")
        print("features =",features)
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features)
        # Calcul de la norme L2 sur la dernière dimension
        norm = torch.norm(features, p=2, dim=-1, keepdim=True)
        # Division par la norme pour obtenir un vecteur unitaire
        return features / norm

    @classmethod
    def embed_batch(cls, images_bytes: List[bytes]) -> List[List[float]]:
        """
        Génère des embeddings pour une liste d'images (format bytes).
        
        Args:
            images_bytes: Liste de contenus d'images en octets.
            
        Returns:
            Une liste de vecteurs (listes de floats) de dimension 512.
        """
        model = cls.get_model()
        processor = cls.get_processor()

        # Conversion des octets en objets PIL Image et passage en mode RGB
        images = [
            Image.open(io.BytesIO(img)).convert("RGB")
            for img in images_bytes
        ]

        # Pré-traitement des images pour le modèle CLIP
        inputs = processor(images=images, return_tensors="pt", padding=True).to(cls._device)

        # Désactivation du calcul du gradient pour l'inférence (gain de mémoire et de temps)
        with torch.no_grad():
            # Extraction des caractéristiques visuelles
            features = model.get_image_features(**inputs)

        # Normalisation des embeddings
        features = cls._normalize(features)

        # Transfert vers le CPU et conversion en liste Python
        return features.cpu().numpy().tolist()

    @classmethod
    def embed_text(cls, text: str) -> List[float]:
        """
        Génère un embedding pour un texte donné à l'aide de CLIP.
        Permet la recherche sémantique "Texte vers Image".
        
        Args:
            text: Le texte à encoder.
            
        Returns:
            Un vecteur (liste de floats) de dimension 512.
        """
        model = cls.get_model()
        processor = cls.get_processor()

        # Pré-traitement du texte pour le modèle CLIP
        inputs = processor(text=[text], return_tensors="pt", padding=True).to(cls._device)

        # Désactivation du calcul du gradient pour l'inférence
        with torch.inference_mode():
            # Extraction des caractéristiques textuelles
            features = model.get_text_features(**inputs)

        # Normalisation de l'embedding pour la similarité cosinus
        features = cls._normalize(features)

        # Transfert vers le CPU et conversion en liste Python
        return features.cpu().numpy().tolist()[0]


    @classmethod
    def embed(cls, image_bytes: bytes) -> List[float]:
        """
        Génère un embedding pour une seule image.
        """
        return cls.embed_batch([image_bytes])[0]