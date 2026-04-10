import certifi
import sentry_sdk
from fastapi import FastAPI
from fastapi.routing import APIRoute
from sqlalchemy import text
from sqlmodel import Session
from starlette.middleware.cors import CORSMiddleware

from app.api.main import api_router
from app.core.config import settings
from app.core.db import engine, init_db


def custom_generate_unique_id(route: APIRoute) -> str:
    return f"{route.tags[0]}-{route.name}"


if settings.SENTRY_DSN and settings.ENVIRONMENT != "local":
    sentry_sdk.init(dsn=str(settings.SENTRY_DSN), enable_tracing=True)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    generate_unique_id_function=custom_generate_unique_id,
)

# Set all CORS enabled origins
if settings.all_cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.all_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(api_router, prefix=settings.API_V1_STR)

@app.on_event("startup")
def on_startup():
    print("🚀 Initializing database...")
    print(certifi.where())
    try:
        # 🔥 FORCER une connexion
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        # 🔥 init data
        with Session(engine) as session:
            init_db(session)

        print("✅ DB OK")
    except Exception as e:
        print(f"❌ DB init error: {e}")
        raise