"""SQLAlchemy models for synthetic manufacturing quality data."""

from datetime import date as DateValue

from sqlalchemy import CheckConstraint, Date, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for RootCause ORM models."""


class ProductionDefect(Base):
    """A single synthetic production defect observation."""

    __tablename__ = "production_defects"
    __table_args__ = (CheckConstraint("severity BETWEEN 1 AND 5", name="ck_defect_severity"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    plant: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    line: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    part: Mapped[str] = mapped_column(String(100), nullable=False)
    defect_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    severity: Mapped[int] = mapped_column(Integer, nullable=False)
    date: Mapped[DateValue] = mapped_column(Date, nullable=False, index=True)


class DowntimeEvent(Base):
    """A synthetic production-line downtime observation."""

    __tablename__ = "downtime_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    plant: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    line: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    reason: Mapped[str] = mapped_column(String(160), nullable=False)
    minutes: Mapped[int] = mapped_column(Integer, nullable=False)
    date: Mapped[DateValue] = mapped_column(Date, nullable=False, index=True)
