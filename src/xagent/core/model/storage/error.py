class StorageError(Exception):
    """Base class for storage-related errors."""

    pass


class ModelHubError(StorageError):
    """Base class for model hub storage errors."""


class ModelNotFoundError(ModelHubError, ValueError):
    """Raised when a model cannot be found in the model hub."""

    model_id: str

    def __init__(self, model_id: str):
        self.model_id = model_id
        super().__init__(f"Model not found: {model_id}")


class InvalidModelRecordError(ModelHubError):
    """Raised when a persisted model hub row cannot be converted."""


class UnsupportedModelCategoryError(InvalidModelRecordError):
    """Raised when a model hub row has an unsupported category."""

    model_id: str
    category: object

    def __init__(self, model_id: str, category: object):
        self.model_id = model_id
        self.category = category
        super().__init__(f"Unknown model category for {model_id}: {category}")


class StorageWriteError(StorageError):
    """Raised when writing to storage fails."""

    path: str
    reason: str

    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        self.message = f"Failed to write to {path}: {reason}"


class StorageReadError(StorageError):
    """Raised when reading from storage fails."""

    path: str
    reason: str

    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        self.message = f"Failed to read from {path}: {reason}"


class InvalidModelError(StorageError):
    """Raised when parsing model from storage fails."""

    path: str
    reason: str

    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        self.message = f"Failed to parse model from {path}: {reason}"
