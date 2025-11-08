"""fix_boolean_string_values

Revision ID: 708ae925a923
Revises: d4dda322022e
Create Date: 2025-11-08 15:56:23.782526

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '708ae925a923'
down_revision: Union[str, Sequence[str], None] = 'd4dda322022e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema - Converte i valori stringa 'true'/'false' in booleani 0/1."""
    # Converti stringhe 'true'/'false' in 1/0 per leagues.is_favorite
    op.execute("UPDATE leagues SET is_favorite = 1 WHERE is_favorite = 'true'")
    op.execute("UPDATE leagues SET is_favorite = 0 WHERE is_favorite = 'false'")
    
    # Converti stringhe 'true'/'false' in 1/0 per teams.is_top
    op.execute("UPDATE teams SET is_top = 1 WHERE is_top = 'true'")
    op.execute("UPDATE teams SET is_top = 0 WHERE is_top = 'false'")


def downgrade() -> None:
    """Downgrade schema - Non necessario per questa fix."""
    pass
