import uuid
from datetime import datetime, timezone

from pydantic import EmailStr
from sqlalchemy import JSON, Column, DateTime
from sqlmodel import Field, Relationship, SQLModel
from pgvector.sqlalchemy import Vector

def get_datetime_utc() -> datetime:
    return datetime.now(timezone.utc)


# Shared properties
class UserBase(SQLModel):
    email: EmailStr = Field(unique=True, index=True, max_length=255)
    is_active: bool = True
    is_superuser: bool = False
    full_name: str | None = Field(default=None, max_length=255)


# Properties to receive via API on creation
class UserCreate(UserBase):
    password: str = Field(min_length=8, max_length=128)


class UserRegister(SQLModel):
    email: EmailStr = Field(max_length=255)
    password: str = Field(min_length=8, max_length=128)
    full_name: str | None = Field(default=None, max_length=255)


# Properties to receive via API on update, all are optional
class UserUpdate(UserBase):
    email: EmailStr | None = Field(default=None, max_length=255)  # type: ignore[assignment]
    password: str | None = Field(default=None, min_length=8, max_length=128)


class UserUpdateMe(SQLModel):
    full_name: str | None = Field(default=None, max_length=255)
    email: EmailStr | None = Field(default=None, max_length=255)


class UpdatePassword(SQLModel):
    current_password: str = Field(min_length=8, max_length=128)
    new_password: str = Field(min_length=8, max_length=128)


# Database model, database table inferred from class name
class User(UserBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    hashed_password: str
    created_at: datetime | None = Field(
        default_factory=get_datetime_utc,
        sa_type=DateTime(timezone=True),  # type: ignore
    )
    items: list["Item"] = Relationship(back_populates="owner", cascade_delete=True)


# Properties to return via API, id is always required
class UserPublic(UserBase):
    id: uuid.UUID
    created_at: datetime | None = None


class UsersPublic(SQLModel):
    data: list[UserPublic]
    count: int


# Shared properties
class ItemBase(SQLModel):
    title: str = Field(min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=255)


# Properties to receive on item creation
class ItemCreate(ItemBase):
    pass


# Properties to receive on item update
class ItemUpdate(ItemBase):
    title: str | None = Field(default=None, min_length=1, max_length=255)  # type: ignore[assignment]


# Database model, database table inferred from class name
class Item(ItemBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    created_at: datetime | None = Field(
        default_factory=get_datetime_utc,
        sa_type=DateTime(timezone=True),  # type: ignore
    )
    owner_id: uuid.UUID = Field(
        foreign_key="user.id", nullable=False, ondelete="CASCADE"
    )
    owner: User | None = Relationship(back_populates="items")


# Properties to return via API, id is always required
class ItemPublic(ItemBase):
    id: uuid.UUID
    owner_id: uuid.UUID
    created_at: datetime | None = None


class ItemsPublic(SQLModel):
    data: list[ItemPublic]
    count: int


class ImagesPublic(SQLModel):
    data: list[ImageUploadPublic]
    count: int


# Generic message
class Message(SQLModel):
    message: str


# JSON payload containing access token
class Token(SQLModel):
    access_token: str
    token_type: str = "bearer"


# Contents of JWT token
class TokenPayload(SQLModel):
    sub: str | None = None


class NewPassword(SQLModel):
    token: str
    new_password: str = Field(min_length=8, max_length=128)


# Modèle ImageUpload pour stocker les images et leurs vecteurs d'embedding
class ImageUploadBase(SQLModel):
    """
    Base commune pour le téléchargement d'images.
    """
    description: str | None = Field(default=None, max_length=1000)


class ImageUploadCreate(ImageUploadBase):
    """
    Schéma pour la création d'un enregistrement d'image via l'API.
    """
    pass


class ImageUpload(ImageUploadBase, table=True):
    """
    Modèle de base de données (Table) pour les images téléchargées.
    Inclut un support pour les vecteurs d'embedding via pgvector.
    """
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    # URL ou chemin vers l'image stockée
    image_url: str = Field(max_length=1000)
    # Date de création avec gestion du fuseau horaire UTC
    created_at: datetime = Field(
        default_factory=get_datetime_utc,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    # Colonne vectorielle pour la recherche par similarité (dimension 512 pour CLIP)
    # Utilise l'extension pgvector de PostgreSQL.
    embedding: list[float] | None = Field(
        default=None,
        sa_column=Column(Vector(512))  # Dimension adaptée pour CLIP (ViT-B/32)
    )


class ImageUploadPublic(ImageUploadBase):
    """
    Modèle pour retourner les métadonnées d'une image sans le vecteur brut.
    """
    id: uuid.UUID
    image_url: str
    created_at: datetime


class ImageSearchResponse(ImageUploadPublic):
    """
    Réponse de recherche incluant le score de similarité (distance).
    """
    similarity_score: float


class ImageSearchListResponse(SQLModel):
    """
    Liste de résultats de recherche.
    """
    data: list[ImageSearchResponse]
    count: int