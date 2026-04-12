import uuid
from typing import Any

from fastapi import APIRouter, File, Form, UploadFile
from sqlalchemy import select, func
from sqlmodel import col

from app.api.deps import CurrentUser, SessionDep
from app.api.routes.chatcontroller import translate_chat_query
from app.models import ImageUpload, ImageSearchListResponse, ImageSearchResponse, ImagesPublic, ImageUploadPublic
from app.service.image_embedding import ImageEmbedding

router = APIRouter(prefix="/images", tags=["images"])


@router.get("/all", response_model=ImagesPublic)
def read_image(session: SessionDep, current_user: CurrentUser, skip: int = 0, limit: int = 100) -> Any:
    """
    Récupère la liste des images téléchargées avec pagination.
    """
    if current_user.is_superuser:
        # Les superutilisateurs voient tout
        count_statement = select(func.count()).select_from(ImageUpload)
        count = session.exec(count_statement).one()
        statement = (
            select(ImageUpload).order_by(col(ImageUpload.created_at).desc()).offset(skip).limit(limit)
        )
        items = session.exec(statement).all()
    else:
        # Pour l'instant, tout le monde voit tout (pas de filtrage par propriétaire encore implémenté)
        count_statement = (
            select(func.count())
            .select_from(ImageUpload)
        )
        count = session.exec(count_statement).one()
        statement = (
            select(ImageUpload)
            .order_by(col(ImageUpload.created_at).desc())
            .offset(skip)
            .limit(limit)
        )
        items = session.exec(statement).all()
    
    # Retourne les données avec le compte total (utile pour le frontend)
    return ImagesPublic(data=items, count=count)


@router.post("/", response_model=ImageUploadPublic)
async def upload_image(
        *,
        session: SessionDep,
        current_user: CurrentUser,
        file: UploadFile = File(...),
        description: str | None = Form(None),
) -> Any:
    """
    Télécharge une image, génère son embedding vectoriel via CLIP et l'enregistre en base de données.
    """
    # En production, le fichier serait sauvegardé sur un stockage persistant (S3, volume, etc.)
    # Ici, nous générons une URL fictive pour l'exemple
    image_url = f"/uploads/{uuid.uuid4()}_{file.filename}"
    
    # Lecture du contenu binaire du fichier téléchargé
    contents = await file.read()
    
    # Génération du vecteur (embedding) de dimension 512 à l'aide du service AI CLIP
    embedding = ImageEmbedding.embed(contents)

    # Création de l'entrée en base de données
    image_upload = ImageUpload(
        description=description,
        image_url=image_url,
        embedding=embedding,
    )
    session.add(image_upload)
    session.commit()
    session.refresh(image_upload)
    
    return image_upload


@router.get("/search", response_model=ImageSearchListResponse)
def search_images(
        *,
        session: SessionDep,
        query: str,
        limit: int = 10,
) -> Any:
    """
    Recherche sémantique d'images à partir d'une requête textuelle.
    Le texte est transformé en vecteur par CLIP, puis comparé aux embeddings en base de données via pgvector.
    """
    # 1. Générer l'embedding pour la requête textuelle

    translate_query = translate_chat_query(query)
    query_embedding = ImageEmbedding.embed_text(translate_query["content"])
    distance = ImageUpload.embedding.cosine_distance(query_embedding)

    # 2. Effectuer la recherche par similarité cosinus (ou distance L2, ici on utilise l'opérateur de pgvector)
    # Dans SQLAlchemy/SQLModel avec pgvector, on utilise .cosine_distance()
    statement = (
        select(
            ImageUpload,
            distance.label("distance")
        ).where(distance < 0.3)
        .order_by("distance")
        .limit(limit)
    )
    
    results = session.exec(statement).all()

    # 3. Formater la réponse
    data = []
    for image, distance in results:
        # Conversion de la distance en score de similarité (optionnel, ici on retourne juste la distance nommée similarity_score par convention inverse)
        # Plus la distance est petite, plus c'est similaire.
        data.append(
            ImageSearchResponse(
                **image.model_dump(),
                similarity_score=1.0 - float(distance) if distance is not None else 0.0
            )
        )

    return ImageSearchListResponse(data=data, count=len(data))
