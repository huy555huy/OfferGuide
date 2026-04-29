"""Local-first memory: SQLite for relational data + sqlite-vec for embeddings."""

from .db import Store
from .vec import attach_vec, init_vec_schema, vec_version

__all__ = ["Store", "attach_vec", "init_vec_schema", "vec_version"]
