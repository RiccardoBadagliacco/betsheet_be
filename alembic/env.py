"""
Minimal Alembic env.py scaffold. Configure alembic.ini to point to your DB or edit the run_migrations_online function.
"""
from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.core.settings import settings
from app.db.models import Base

# Prefer using settings to obtain the DB URL when running alembic from the project
SQLALCHEMY_DATABASE_URL = settings.SQLALCHEMY_DATABASE_URL

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_online():
    # If SQLALCHEMY URL provided in alembic.ini, it will be used; otherwise fall back to settings
    cfg = config.get_section(config.config_ini_section)
    if not cfg.get('sqlalchemy.url'):
        cfg['sqlalchemy.url'] = SQLALCHEMY_DATABASE_URL

    connectable = engine_from_config(
        cfg,
        prefix='sqlalchemy.',
        poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    raise RuntimeError('Offline mode not supported in this scaffold')
else:
    run_migrations_online()
