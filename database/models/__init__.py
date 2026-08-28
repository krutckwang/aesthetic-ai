"""SQLAlchemy ORM models for the aesthetic-ai database."""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class ConsentTier(enum.IntEnum):
    CONFIRMED = 1
    LIKELY = 2
    UNCERTAIN = 3


class OrderingConfidence(str, enum.Enum):
    HIGH = "HIGH"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"


class Image(Base):
    __tablename__ = "image"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    source_url: Mapped[str] = mapped_column(String(2048), nullable=False, unique=True)
    domain: Mapped[str] = mapped_column(String(256), nullable=False)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    file_size_bytes: Mapped[int | None] = mapped_column(Integer)
    language: Mapped[str] = mapped_column(String(10), default="en")
    collected_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    consent_tier: Mapped[int] = mapped_column(Integer, nullable=False)

    quality_score: Mapped["QualityScore | None"] = relationship(back_populates="image", uselist=False)
    consent_record: Mapped["ConsentRecord | None"] = relationship(back_populates="image", uselist=False)
    landmarks: Mapped[list["Landmark"]] = relationship(back_populates="image")


class ImagePair(Base):
    __tablename__ = "image_pair"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    before_image_id: Mapped[int] = mapped_column(ForeignKey("image.id"), nullable=False)
    after_image_id: Mapped[int] = mapped_column(ForeignKey("image.id"), nullable=False)
    layer1_score: Mapped[float | None] = mapped_column(Float)
    layer2_score: Mapped[float | None] = mapped_column(Float)
    layer3_score: Mapped[float | None] = mapped_column(Float)
    pair_confidence: Mapped[float | None] = mapped_column(Float)
    ordering_confidence: Mapped[str] = mapped_column(String(10), default=OrderingConfidence.UNKNOWN.value)
    in_training_set: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (UniqueConstraint("before_image_id", "after_image_id", name="uq_pair"),)

    before_image: Mapped["Image"] = relationship(foreign_keys=[before_image_id])
    after_image: Mapped["Image"] = relationship(foreign_keys=[after_image_id])
    treatment_label: Mapped["TreatmentLabel | None"] = relationship(back_populates="pair", uselist=False)
    zone_labels: Mapped[list["ZoneLabel"]] = relationship(back_populates="pair")
    source_metadata: Mapped["SourceMetadata | None"] = relationship(back_populates="pair", uselist=False)


class TreatmentLabel(Base):
    __tablename__ = "treatment_label"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    pair_id: Mapped[int] = mapped_column(ForeignKey("image_pair.id"), nullable=False, unique=True)
    treatment_category: Mapped[str | None] = mapped_column(String(64))
    treatment_brand: Mapped[str | None] = mapped_column(String(128))
    confidence: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(16), default="auto")

    pair: Mapped["ImagePair"] = relationship(back_populates="treatment_label")


class ZoneLabel(Base):
    __tablename__ = "zone_label"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    pair_id: Mapped[int] = mapped_column(ForeignKey("image_pair.id"), nullable=False)
    zone_code: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(16), default="auto")

    pair: Mapped["ImagePair"] = relationship(back_populates="zone_labels")


class QualityScore(Base):
    __tablename__ = "quality_score"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("image.id"), nullable=False, unique=True)
    blur_score: Mapped[float | None] = mapped_column(Float)
    lighting_score: Mapped[float | None] = mapped_column(Float)
    resolution_pass: Mapped[bool | None] = mapped_column(Boolean)
    overall_grade: Mapped[str | None] = mapped_column(String(8))

    image: Mapped["Image"] = relationship(back_populates="quality_score")


class Landmark(Base):
    __tablename__ = "landmark"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("image.id"), nullable=False)
    landmark_index: Mapped[int] = mapped_column(Integer, nullable=False)
    x: Mapped[float] = mapped_column(Float, nullable=False)
    y: Mapped[float] = mapped_column(Float, nullable=False)
    z: Mapped[float] = mapped_column(Float, nullable=False)

    image: Mapped["Image"] = relationship(back_populates="landmarks")

    __table_args__ = (UniqueConstraint("image_id", "landmark_index", name="uq_landmark"),)


class ConsentRecord(Base):
    __tablename__ = "consent_record"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("image.id"), nullable=False, unique=True)
    consent_tier: Mapped[int] = mapped_column(Integer, nullable=False)
    signals_found: Mapped[str | None] = mapped_column(Text)
    assessed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    image: Mapped["Image"] = relationship(back_populates="consent_record")


class Quarantine(Base):
    """Tier 3 uncertain-consent images. Never joined with training queries."""

    __tablename__ = "quarantine"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("image.id"), nullable=False, unique=True)
    reason: Mapped[str] = mapped_column(String(256), nullable=False)
    source_url: Mapped[str] = mapped_column(String(2048))
    assessed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class SourceMetadata(Base):
    __tablename__ = "source_metadata"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    pair_id: Mapped[int] = mapped_column(ForeignKey("image_pair.id"), nullable=False, unique=True)
    practitioner_name: Mapped[str | None] = mapped_column(String(256))
    clinic_name: Mapped[str | None] = mapped_column(String(256))
    date_posted: Mapped[str | None] = mapped_column(String(64))
    geographic_region: Mapped[str | None] = mapped_column(String(128))
    language: Mapped[str | None] = mapped_column(String(10))
    source_name: Mapped[str | None] = mapped_column(String(64))
    raw_metadata: Mapped[str | None] = mapped_column(Text)

    pair: Mapped["ImagePair"] = relationship(back_populates="source_metadata")
