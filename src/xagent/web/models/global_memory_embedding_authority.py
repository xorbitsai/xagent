from sqlalchemy import Boolean, CheckConstraint, Column, DateTime, Integer, String, Text
from sqlalchemy.sql import func

from .database import Base


class GlobalMemoryEmbeddingAuthority(Base):  # type: ignore[no-any-unimported]
    __tablename__ = "global_memory_embedding_authority"
    __table_args__ = (
        CheckConstraint(
            "authority_key = 'global'", name="ck_global_memory_authority_key"
        ),
        # Enforced at rest too: a row written around the service can never
        # present a personal or unconsented credential as the authority.
        CheckConstraint(
            "credential_source IN ('application_owned', 'organization_owned')",
            name="ck_global_memory_authority_credential_source",
        ),
        CheckConstraint(
            "global_sharing_consent", name="ck_global_memory_authority_consent"
        ),
    )

    authority_key = Column(String(32), primary_key=True)
    model_provider = Column(String(50), nullable=False)
    model_name = Column(String(100), nullable=False)
    base_url = Column(String(500), nullable=False)
    dimension = Column(Integer, nullable=False)
    instruct = Column(Text, nullable=True)
    max_retries = Column(Integer, nullable=False, default=10, server_default="10")
    api_key_encrypted = Column(Text, nullable=False)
    credential_verifier = Column(String(96), nullable=False)
    credential_source = Column(String(32), nullable=False)
    global_sharing_consent = Column(Boolean, nullable=False)
    configured_by_actor_subject = Column(String(64), nullable=False)
    consented_by_actor_subject = Column(String(64), nullable=False)
    consented_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
