"""add imageupload table with vector column

Revision ID: 202604071200
Revises: fe56fa70289e
Create Date: 2026-04-07 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision = '202604071200'
down_revision = 'fe56fa70289e'
branch_labels = None
depends_on = None



def upgrade():
    # Enable extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')

    op.create_table(
        'imageupload',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('description', sa.String(length=1000)),
        sa.Column('image_url', sa.String(length=1000), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True)),
        sa.Column('embedding', Vector(512)),
        sa.PrimaryKeyConstraint('id')
    )

    # Index optimisé
    op.execute(
        "CREATE INDEX image_embedding_idx ON imageupload USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )
    op.execute(
        "ANALYZE imageupload"
    )


def downgrade():
    op.drop_table('imageupload')
    # We might not want to drop the extension if other tables use it
    # op.execute('DROP EXTENSION vector')
