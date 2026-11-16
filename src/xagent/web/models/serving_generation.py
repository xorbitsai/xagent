"""The single-row record naming the generation that may claim singleton work."""

from sqlalchemy import CheckConstraint, Column, DateTime, Integer, String
from sqlalchemy.sql import func

from .database import Base


class ServingGeneration(Base):  # type: ignore
    __tablename__ = "serving_generation"
    __table_args__ = (
        CheckConstraint("id = 1", name="ck_serving_generation_single_row"),
    )

    id = Column(Integer, primary_key=True, nullable=False)
    generation = Column(String(255), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
