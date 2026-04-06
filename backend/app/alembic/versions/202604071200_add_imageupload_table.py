"""add imageupload table with vector column

Revision ID: 202604071200
Revises: fe56fa70289e
Create Date: 2026-04-07 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


# revision identifiers, used by Alembic.
revision = '202604071200'
down_revision = 'fe56fa70289e'
branch_labels = None
depends_on = None


def upgrade():
    # Enable pgvector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    
    op.create_table('imageupload',
        sa.Column('id', sqlmodel.sql.sqltypes.GUID(), nullable=False),
        sa.Column('description', sa.String(length=1000), nullable=True),
        sa.Column('image_url', sa.String(length=1000), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('embedding', sa.NullType(), nullable=True), # Placeholder for Vector
        sa.PrimaryKeyConstraint('id')
    )
    # Manually add the vector column with correct type
    op.execute('ALTER TABLE imageupload ALTER COLUMN embedding TYPE vector(3) USING embedding::vector(3)')

def downgrade():
    op.drop_table('imageupload')
    # We might not want to drop the extension if other tables use it
    # op.execute('DROP EXTENSION vector')
