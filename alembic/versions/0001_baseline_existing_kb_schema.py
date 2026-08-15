"""empty message

Revision ID: 0001_baseline
Revises:
Create Date: 2026-08-15

Baseline for existing Neon kb_* schema.
No DDL: vector extension, kb_documents, kb_chunks, and HNSW index
already exist in the target database.
"""

from typing import Sequence, Union

revision: str = "0001_baseline"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Schema already present on Neon (pgvector + kb_* tables).
    # Keep this revision as the Alembic head so future diffs can be applied.
    pass


def downgrade() -> None:
    pass
