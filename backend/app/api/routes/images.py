import uuid
from typing import Any

import ollama
from fastapi import APIRouter, File, Form, UploadFile
from sqlalchemy import select, func
from sqlmodel import col

from app.api.deps import CurrentUser, SessionDep
from app.models import ImageUpload

router = APIRouter(prefix="/images", tags=["images"])

response = ollama.embeddings(
    model="nomic-embed-text",
    prompt="a cat on a table"
)

embedding = response["embedding"]


@router.get("/all", response_model=ImageUpload)
def read_image(session: SessionDep, current_user: CurrentUser, skip: int = 0, limit: int = 100) -> Any:
    if current_user.is_superuser:
        count_statement = select(func.count()).select_from(ImageUpload)
        count = session.exec(count_statement).one()
        statement = (
            select(ImageUpload).order_by(col(ImageUpload.created_at).desc()).offset(skip).limit(limit)
        )
        items = session.exec(statement).all()
    else:
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
    items_public = [ImageUpload.model_validate(item) for item in items]
    return ImageUpload(data=items_public, count=count)


@router.post("/", response_model=ImageUpload)
def upload_image(
        *,
        session: SessionDep,
        current_user: CurrentUser,
        file: UploadFile = File(...),
        description: str | None = Form(None),
) -> Any:
    """
    Upload an image and store its metadata and a vector embedding.
    """
    # In a real scenario, we would save the file to a storage (S3, local disk, etc.)
    # and generate an embedding using a model (e.g., CLIP, ResNet, etc.)

    # Placeholder for image URL (assuming local storage or just a name for now)
    image_url = f"/uploads/{uuid.uuid4()}_{file.filename}"

    # Placeholder for vector embedding (e.g., dimension 3 for this example)
    # The actual storage will depend on pgvector being installed and the column being Vector type.
    placeholder_embedding = [0.1, 0.2, 0.3]

    image_upload = ImageUpload(
        description=description,
        image_url=image_url,
        embedding=placeholder_embedding,
    )
    session.add(image_upload)
    session.commit()
    session.refresh(image_upload)
    return image_upload
