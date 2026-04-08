from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from pwdlib import PasswordHash
from pwdlib.hashers.argon2 import Argon2Hasher
from pwdlib.hashers.bcrypt import BcryptHasher

from app.core.config import settings

if settings.ENVIRONMENT == "local":
    # En mode local, on réduit la complexité pour accélérer le développement
    argon2_hasher = Argon2Hasher(time_cost=1, memory_cost=1024, parallelism=1)
    bcrypt_hasher = BcryptHasher(rounds=4)
else:
    # Paramètres sécurisés par défaut pour staging/production
    argon2_hasher = Argon2Hasher()
    bcrypt_hasher = BcryptHasher()

password_hash = PasswordHash(
    (
        argon2_hasher,
        bcrypt_hasher,
    )
)


ALGORITHM = "HS256"


def create_access_token(subject: str | Any, expires_delta: timedelta) -> str:
    expire = datetime.now(timezone.utc) + expires_delta
    to_encode = {"exp": expire, "sub": str(subject)}
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_password(
    plain_password: str, hashed_password: str
) -> tuple[bool, str | None]:
    return password_hash.verify_and_update(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return password_hash.hash(password)
