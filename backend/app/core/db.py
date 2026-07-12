import logging
import time

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlmodel import Session, create_engine, select

from app import crud
from app.core.config import settings
from app.models import User, UserCreate

logger = logging.getLogger("app.db")

engine = create_engine(
    str(settings.SQLALCHEMY_DATABASE_URI),
    echo=settings.SQLALCHEMY_ECHO,
    # pré-ping : évite les erreurs sur connexions coupées sans surcoût notable.
    pool_pre_ping=True,
    # recycle des connexions au bout d'une heure (évite les timeouts serveur).
    pool_recycle=3600,
)


def _register_query_timing_listeners() -> None:
    """Instrumentation de timing par requête.

    Coûteux (log de chaque statement + paramètres) : réservé au debug et donc
    branché uniquement quand ``SQLALCHEMY_ECHO`` est actif. C'était la cause de la
    lenteur au démarrage (migrations + init_db générant des milliers de logs).
    """

    @event.listens_for(Engine, "before_cursor_execute")
    def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):  # type: ignore[no-untyped-def]
        context._query_start_time = time.time()

    @event.listens_for(Engine, "after_cursor_execute")
    def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):  # type: ignore[no-untyped-def]
        total = time.time() - context._query_start_time
        logger.debug("Query (%.4fs): %s | params=%s", total, statement, parameters)


if settings.SQLALCHEMY_ECHO:
    logging.getLogger("sqlalchemy.engine").setLevel(logging.DEBUG)
    _register_query_timing_listeners()


# make sure all SQLModel models are imported (app.models) before initializing DB
# otherwise, SQLModel might fail to initialize relationships properly
# for more details: https://github.com/fastapi/full-stack-fastapi-template/issues/28


def init_db(session: Session) -> None:
    # Tables should be created with Alembic migrations
    # But if you don't want to use migrations, create
    # the tables un-commenting the next lines
    # from sqlmodel import SQLModel

    # This works because the models are already imported and registered from app.models
    # SQLModel.metadata.create_all(engine)

    user = session.exec(
        select(User).where(User.email == settings.FIRST_SUPERUSER)
    ).first()
    if not user:
        user_in = UserCreate(
            email=settings.FIRST_SUPERUSER,
            password=settings.FIRST_SUPERUSER_PASSWORD,
            is_superuser=True,
        )
        user = crud.create_user(session=session, user_create=user_in)
