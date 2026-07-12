import io
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.config import settings
from tests.utils.image import create_random_image


def test_upload_image(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    # On mocke ImageEmbedding.embed pour éviter de charger le modèle CLIP pendant les tests
    with patch("app.api.routes.images.ImageEmbedding.embed") as mock_embed:
        mock_embed.return_value = [0.1] * 512
        
        file_content = b"fake image content"
        files = {"file": ("test.jpg", io.BytesIO(file_content), "image/jpeg")}
        data = {"description": "Test upload description"}
        
        response = client.post(
            f"{settings.API_V1_STR}/images/",
            headers=superuser_token_headers,
            data=data,
            files=files,
        )
        
        assert response.status_code == 200
        content = response.json()
        assert content["description"] == data["description"]
        assert "id" in content
        assert "image_url" in content
        assert "created_at" in content
        # L'embedding ne doit PAS être dans la réponse publique
        assert "embedding" not in content


def test_read_images(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    create_random_image(db)
    create_random_image(db)
    
    response = client.get(
        f"{settings.API_V1_STR}/images/all",
        headers=superuser_token_headers,
    )
    
    assert response.status_code == 200
    content = response.json()
    assert "data" in content
    assert "count" in content
    assert len(content["data"]) >= 2
    # Vérifier qu'un élément de la liste n'a pas d'embedding
    assert "embedding" not in content["data"][0]


def test_search_images(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    create_random_image(db, description="Searchable image")
    
    # On mocke ImageEmbedding.embed_text
    with patch("app.api.routes.images.ImageEmbedding.embed_text") as mock_embed_text:
        mock_embed_text.return_value = [0.1] * 512
        
        response = client.get(
            f"{settings.API_V1_STR}/images/search",
            headers=superuser_token_headers,
            params={"query": "search query"},
        )
        
        assert response.status_code == 200
        content = response.json()
        assert "data" in content
        assert "count" in content
        assert len(content["data"]) > 0
        assert "similarity_score" in content["data"][0]
        # L'embedding ne doit PAS être dans la réponse de recherche
        assert "embedding" not in content["data"][0]
