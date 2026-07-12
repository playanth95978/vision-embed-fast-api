"""add RAG vector tables (vector_docs / vector_jira / vector_pdf / vector_code)

Schéma compatible Spring AI PgVectorStore : id (uuid) · content (text) · metadata (jsonb)
· embedding (vector). Index HNSW cosine sur chaque table. Index BM25 ParadeDB créés
uniquement si l'extension ``pg_search`` est présente (sinon la recherche full-text
dégrade proprement côté application).

Revision ID: 202607131300
Revises: 202604071200
Create Date: 2026-07-13 13:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision = "202607131300"
down_revision = "202604071200"
branch_labels = None
depends_on = None

# (table, dimension d'embedding)
_TABLES: list[tuple[str, int]] = [
    ("vector_docs", 768),
    ("vector_jira", 768),
    ("vector_pdf", 768),
    ("vector_code", 1024),
]


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    for table, dim in _TABLES:
        op.create_table(
            table,
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("content", sa.Text(), nullable=False),
            sa.Column("metadata", JSONB(), nullable=False),
            sa.Column("embedding", Vector(dim)),
            sa.PrimaryKeyConstraint("id"),
        )
        # Index vectoriel HNSW (similarité cosine), comme le backend Java.
        op.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{table}_hnsw "
            f"ON {table} USING hnsw (embedding vector_cosine_ops)"
        )
        # Index full-text BM25 (ParadeDB) — créé seulement si l'extension est installée.
        op.execute(
            f"""
            DO $$
            BEGIN
                IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_search') THEN
                    EXECUTE 'CREATE INDEX IF NOT EXISTS idx_{table}_bm25 '
                         || 'ON {table} USING bm25 (id, content) WITH (key_field=''id'')';
                END IF;
            END $$;
            """
        )


def downgrade() -> None:
    for table, _ in _TABLES:
        op.drop_table(table)
    # On ne supprime pas l'extension vector : d'autres tables (imageupload) l'utilisent.
