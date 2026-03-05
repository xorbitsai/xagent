from typing import Any

from sqlalchemy.orm import Session

from xagent.core.model.model import (
    ChatModelConfig,
    EmbeddingModelConfig,
    ImageModelConfig,
    ModelConfig,
    RerankModelConfig,
    VectorDBConfig,
    VectorDBType,
)


class SQLAlchemyModelHub:
    """SQLAlchemy implementation of standalone model storage"""

    def __init__(self, db_session: Session, model_class: Any):
        """
        Initialize SQLAlchemy model storage.

        Args:
            db_session: SQLAlchemy database session
            model_class: Model ORM class
        """
        self.db = db_session
        self.Model = model_class

    @staticmethod
    def _normalize_vector_db_type(raw_value: Any) -> str:
        """Normalize vector db type values for backward compatibility."""
        if raw_value is None:
            return "lancedb"
        normalized = str(getattr(raw_value, "value", raw_value)).strip().lower()
        if not normalized:
            return "lancedb"
        legacy_aliases = {
            "weaviate_local": "weaviate",  # backward compatibility
        }
        return legacy_aliases.get(normalized, normalized)

    def _load_vector_db_config(
        self, db_model: Any, common_fields: dict[str, Any]
    ) -> VectorDBConfig:
        """Build VectorDBConfig from DB row; mutates common_fields (pops abilities)."""
        db_type_val = self._normalize_vector_db_type(db_model.model_provider)
        try:
            db_type = VectorDBType(db_type_val) if db_type_val else VectorDBType.LANCEDB
        except ValueError:
            db_type = VectorDBType.LANCEDB
        # VectorDBConfig repurposes abilities column for config dict (ModelConfig.abilities is List[str] elsewhere).
        config_data = common_fields.pop("abilities", {})
        if not isinstance(config_data, dict):
            config_data = {}
        return VectorDBConfig(**common_fields, db_type=db_type, config=config_data)

    def store(self, model: ModelConfig) -> None:
        db_data: dict[str, Any] = {
            "model_id": model.id,
            "model_name": model.model_name,
            "api_key": model.api_key,
            "base_url": model.base_url,
            "abilities": model.abilities,
            "description": model.description,
            "max_retries": model.max_retries,
            "is_active": True,
        }

        if isinstance(model, ChatModelConfig):
            db_data.update(
                {
                    "model_provider": model.model_provider,
                    "temperature": model.default_temperature,
                    "max_tokens": model.default_max_tokens,
                    "category": "llm",
                }
            )
        elif isinstance(model, ImageModelConfig):
            db_data.update(
                {
                    "model_provider": model.model_provider,
                    "max_tokens": model.default_max_tokens,
                    "category": "image",
                }
            )
        elif isinstance(model, EmbeddingModelConfig):
            db_data.update(
                {
                    "model_provider": model.model_provider,
                    "dimension": model.dimension,
                    "category": "embedding",
                }
            )
        elif isinstance(model, RerankModelConfig):
            db_data.update(
                {
                    "model_provider": "none",
                    "category": "rerank",
                }
            )
        elif isinstance(model, VectorDBConfig):
            # VectorDBConfig repurposes abilities column for config dict (ModelConfig.abilities is List[str] elsewhere).
            db_data.update(
                {
                    "model_provider": model.db_type.value,
                    "category": "vector_db",
                    "abilities": model.config,
                }
            )
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")

        db_model = self.Model(**db_data)
        self.db.add(db_model)
        self.db.commit()

    def load(self, model_id: str) -> ModelConfig:
        # Try to find by model_id first, then by model_name
        db_model = (
            self.db.query(self.Model)
            .filter(self.Model.model_id == model_id)
            .filter(self.Model.is_active)
            .first()
        )
        # If not found by model_id, try by model_name
        if not db_model:
            db_model = (
                self.db.query(self.Model)
                .filter(self.Model.model_name == model_id)
                .filter(self.Model.is_active)
                .first()
            )
        if not db_model:
            raise ValueError(f"Model not found: {model_id}")

        common = {
            "id": db_model.model_id,
            "model_name": db_model.model_name,
            "api_key": db_model.api_key,
            "base_url": db_model.base_url,
            "abilities": db_model.abilities,
            "description": db_model.description,
            "max_retries": db_model.max_retries
            if db_model.max_retries is not None
            else 10,
        }

        if db_model.category == "llm":
            return ChatModelConfig(
                **common,
                model_provider=db_model.model_provider,
                default_temperature=db_model.temperature,
                default_max_tokens=db_model.max_tokens,
            )
        elif db_model.category == "image":
            return ImageModelConfig(
                **common,
                model_provider=db_model.model_provider,
                default_max_tokens=db_model.max_tokens,
            )
        elif db_model.category == "embedding":
            return EmbeddingModelConfig(
                **common,
                dimension=db_model.dimension,
                model_provider=db_model.model_provider,
            )
        elif db_model.category == "rerank":
            return RerankModelConfig(**common)
        elif db_model.category == "vector_db":
            return self._load_vector_db_config(db_model, common)
        else:
            raise ValueError(f"Unknown model category: {db_model.category}")

    def list(self) -> dict[str, ModelConfig]:
        db_models = self.db.query(self.Model).filter(self.Model.is_active).all()
        result: dict[str, ModelConfig] = {}

        for db_model in db_models:
            # Common fields for all models
            common_fields = {
                "id": db_model.model_id,
                "model_name": db_model.model_name,
                "api_key": db_model.api_key,
                "base_url": db_model.base_url,
                "abilities": db_model.abilities,
                "description": db_model.description,
                "max_retries": db_model.max_retries
                if db_model.max_retries is not None
                else 10,
            }

            # Create appropriate config based on category
            config: ModelConfig | None = None
            if db_model.category == "llm":
                config = ChatModelConfig(
                    **common_fields,
                    model_provider=db_model.model_provider,
                    default_temperature=db_model.temperature,
                    default_max_tokens=db_model.max_tokens,
                )
            elif db_model.category == "image":
                config = ImageModelConfig(
                    **common_fields,
                    model_provider=db_model.model_provider,
                )
            elif db_model.category == "embedding":
                config = EmbeddingModelConfig(
                    **common_fields,
                    model_provider=db_model.model_provider,
                    dimension=db_model.dimension,
                )
            elif db_model.category == "rerank":
                config = RerankModelConfig(**common_fields)
            elif db_model.category == "vector_db":
                config = self._load_vector_db_config(db_model, common_fields)

            if config:
                result[db_model.model_id] = config

        return result

    def exists(self, model_id: str) -> bool:
        count = (
            self.db.query(self.Model).filter(self.Model.model_id == model_id).count()
        )
        result: bool = count > 0
        return result

    def delete(self, model_id: str) -> None:
        db_model = (
            self.db.query(self.Model).filter(self.Model.model_id == model_id).first()
        )
        if db_model:
            self.db.delete(db_model)
            self.db.commit()
