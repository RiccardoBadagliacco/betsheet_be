"""add_is_favorite_to_league_and_is_top_to_team

Revision ID: d4dda322022e
Revises: 19c7d9d0f37e
Create Date: 2025-11-08 15:01:27.034068

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd4dda322022e'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Aggiungi colonna is_favorite alla tabella leagues
    op.add_column('leagues', sa.Column('is_favorite', sa.Boolean(), nullable=False, server_default='false'))
    
    # Aggiungi colonna is_top alla tabella teams
    op.add_column('teams', sa.Column('is_top', sa.Boolean(), nullable=False, server_default='false'))


def downgrade() -> None:
    """Downgrade schema."""
    # Rimuovi colonna is_favorite dalla tabella leagues
    op.drop_column('leagues', 'is_favorite')
    
    # Rimuovi colonna is_top dalla tabella teams
    op.drop_column('teams', 'is_top')
