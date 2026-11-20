"""Shared Postgres connection pool — reused across calls instead of reconnecting every time.
`register_vector` runs on each new connection so pgvector types work everywhere."""
import os

from dotenv import load_dotenv
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector

load_dotenv(".env.local")

pool = ConnectionPool(
    os.environ["DATABASE_URL"],
    min_size=1, max_size=5,
    configure=register_vector,
    open=True,
)
