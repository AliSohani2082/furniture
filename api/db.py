# api/db.py
from __future__ import annotations

import os
import uuid
from datetime import datetime

from sqlalchemy import ARRAY, JSON, SmallInteger, Text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

DATABASE_URL = os.environ["DATABASE_URL"]

engine = create_async_engine(DATABASE_URL, echo=False)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(Text, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, onupdate=datetime.utcnow)

    # Input
    input_type: Mapped[str] = mapped_column(Text)        # 'url' | 'image'
    input_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    input_s3_key: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Lifecycle
    status: Mapped[str] = mapped_column(Text, default="queued")
    current_stage: Mapped[int] = mapped_column(SmallInteger, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    stage_timings: Mapped[dict] = mapped_column(JSON, default=dict)

    # S1 intelligence metadata
    furniture_category: Mapped[str | None] = mapped_column(Text, nullable=True)
    dimensions_mm: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # S2 crops (parallel arrays)
    crop_s3_keys: Mapped[list[str] | None] = mapped_column(ARRAY(Text), nullable=True)
    mask_s3_keys: Mapped[list[str] | None] = mapped_column(ARRAY(Text), nullable=True)

    # S3–S5 mesh outputs
    mesh_glb_s3_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    mesh_scaled_glb_s3_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    mesh_textured_glb_s3_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    uv_map_s3_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    texture_atlas_s3_key: Mapped[str | None] = mapped_column(Text, nullable=True)

    # S6 render outputs
    render_front_s3_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    render_side_s3_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    render_top_s3_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    render_angled_s3_key: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Modal tracking
    modal_call_id: Mapped[str | None] = mapped_column(Text, nullable=True)


async def get_session():
    async with SessionLocal() as session:
        yield session


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
