"""Initial schema — all tables.

Revision ID: 001
Revises:
Create Date: 2026-08-29
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "image",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("file_path", sa.String(512), nullable=False),
        sa.Column("source_url", sa.String(2048), nullable=False, unique=True),
        sa.Column("domain", sa.String(256), nullable=False),
        sa.Column("width", sa.Integer),
        sa.Column("height", sa.Integer),
        sa.Column("file_size_bytes", sa.Integer),
        sa.Column("language", sa.String(10), server_default="en"),
        sa.Column("collected_at", sa.DateTime, nullable=False),
        sa.Column("consent_tier", sa.Integer, nullable=False),
    )
    op.create_table(
        "image_pair",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("before_image_id", sa.Integer, sa.ForeignKey("image.id"), nullable=False),
        sa.Column("after_image_id", sa.Integer, sa.ForeignKey("image.id"), nullable=False),
        sa.Column("layer1_score", sa.Float),
        sa.Column("layer2_score", sa.Float),
        sa.Column("layer3_score", sa.Float),
        sa.Column("pair_confidence", sa.Float),
        sa.Column("ordering_confidence", sa.String(10), server_default="UNKNOWN"),
        sa.Column("in_training_set", sa.Boolean, server_default="0"),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.UniqueConstraint("before_image_id", "after_image_id", name="uq_pair"),
    )
    op.create_table(
        "treatment_label",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("pair_id", sa.Integer, sa.ForeignKey("image_pair.id"), nullable=False, unique=True),
        sa.Column("treatment_category", sa.String(64)),
        sa.Column("treatment_brand", sa.String(128)),
        sa.Column("confidence", sa.Float),
        sa.Column("source", sa.String(16), server_default="auto"),
    )
    op.create_table(
        "zone_label",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("pair_id", sa.Integer, sa.ForeignKey("image_pair.id"), nullable=False),
        sa.Column("zone_code", sa.String(64), nullable=False),
        sa.Column("confidence", sa.Float),
        sa.Column("source", sa.String(16), server_default="auto"),
    )
    op.create_table(
        "quality_score",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("image_id", sa.Integer, sa.ForeignKey("image.id"), nullable=False, unique=True),
        sa.Column("blur_score", sa.Float),
        sa.Column("lighting_score", sa.Float),
        sa.Column("resolution_pass", sa.Boolean),
        sa.Column("overall_grade", sa.String(8)),
    )
    op.create_table(
        "landmark",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("image_id", sa.Integer, sa.ForeignKey("image.id"), nullable=False),
        sa.Column("landmark_index", sa.Integer, nullable=False),
        sa.Column("x", sa.Float, nullable=False),
        sa.Column("y", sa.Float, nullable=False),
        sa.Column("z", sa.Float, nullable=False),
        sa.UniqueConstraint("image_id", "landmark_index", name="uq_landmark"),
    )
    op.create_table(
        "consent_record",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("image_id", sa.Integer, sa.ForeignKey("image.id"), nullable=False, unique=True),
        sa.Column("consent_tier", sa.Integer, nullable=False),
        sa.Column("signals_found", sa.Text),
        sa.Column("assessed_at", sa.DateTime, nullable=False),
    )
    op.create_table(
        "quarantine",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("image_id", sa.Integer, sa.ForeignKey("image.id"), nullable=False, unique=True),
        sa.Column("reason", sa.String(256), nullable=False),
        sa.Column("source_url", sa.String(2048)),
        sa.Column("assessed_at", sa.DateTime, nullable=False),
    )
    op.create_table(
        "source_metadata",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("pair_id", sa.Integer, sa.ForeignKey("image_pair.id"), nullable=False, unique=True),
        sa.Column("practitioner_name", sa.String(256)),
        sa.Column("clinic_name", sa.String(256)),
        sa.Column("date_posted", sa.String(64)),
        sa.Column("geographic_region", sa.String(128)),
        sa.Column("language", sa.String(10)),
        sa.Column("source_name", sa.String(64)),
        sa.Column("raw_metadata", sa.Text),
    )


def downgrade() -> None:
    op.drop_table("source_metadata")
    op.drop_table("quarantine")
    op.drop_table("consent_record")
    op.drop_table("landmark")
    op.drop_table("quality_score")
    op.drop_table("zone_label")
    op.drop_table("treatment_label")
    op.drop_table("image_pair")
    op.drop_table("image")
