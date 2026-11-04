import inspect
import logging
from collections.abc import Mapping as ABCMapping
from typing import Any, Callable, Mapping, Optional, Type, get_origin

from pydantic import BaseModel, RootModel, ValidationError, create_model

from .base import AbstractBaseTool, ToolCategory, ToolVisibility

# Set up logger
logger = logging.getLogger(__name__)


class FunctionTool(AbstractBaseTool):
    """
    Wrap a Python function into a Tool implementation.

    Automatically generates Pydantic models for input arguments and return type,
    supports synchronous and asynchronous functions.
    Stateless by default.
    """

    # Default category for this tool class - can be overridden per instance
    category: ToolCategory = ToolCategory.OTHER

    def __init__(
        self,
        func: Callable[..., Any],
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
        visibility: Optional[ToolVisibility] = None,
        allow_users: Optional[list[str]] = None,
        read_only: bool = False,
        concurrency_safe: bool = False,
    ):
        self.func = func
        self._name = name or func.__name__
        self._description = description or (
            func.__doc__.strip() if func.__doc__ else ""
        )
        self._tags = tags or []
        self._visibility = visibility or ToolVisibility.PRIVATE
        self._allow_users = allow_users
        # Concurrent-execution semantics (read by AbstractBaseTool.metadata).
        self.read_only = read_only
        self.concurrency_safe = concurrency_safe

        self._args_type = self._build_args_model()
        self._return_type = self._build_return_model()
        self._is_async = inspect.iscoroutinefunction(func)

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def tags(self) -> list[str]:
        return self._tags

    def args_type(self) -> Type[BaseModel]:
        return self._args_type

    def return_type(self) -> Type[BaseModel]:
        return self._return_type

    def state_type(self) -> Optional[Type[BaseModel]]:
        # Stateless by default
        return None

    def is_async(self) -> bool:
        return self._is_async

    def return_value_as_string(self, value: Any) -> str:
        try:
            if isinstance(value, BaseModel):
                return value.json()
            if isinstance(value, dict):
                return str(value)
            return str(value)
        except Exception as e:
            logger.warning(
                f"Failed to convert return value to string for tool '{self._name}': {e}"
            )
            return str(value)

    def _build_args_model(self) -> Type[BaseModel]:
        """
        Create a Pydantic model from the function's signature to validate input arguments.
        """
        sig = inspect.signature(self.func)
        fields: dict[str, Any] = {}
        for name, param in sig.parameters.items():
            # Skip VAR_KEYWORD (**kwargs) and VAR_POSITIONAL (*args) parameters
            # as they don't map to named fields in the tool schema
            if param.kind in (
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            ):
                continue

            annotation = (
                param.annotation if param.annotation != inspect.Parameter.empty else Any
            )
            if param.default != inspect.Parameter.empty:
                fields[name] = (annotation, param.default)
            else:
                fields[name] = (annotation, ...)
        return create_model(f"{self._name}Args", **fields)

    def _build_return_model(self) -> type[BaseModel]:
        """
        Dynamically construct a return model for the tool.
        """
        sig = inspect.signature(self.func)
        ret_ann = sig.return_annotation

        if ret_ann in (inspect.Signature.empty, None):

            class EmptyReturnModel(BaseModel):
                pass

            return EmptyReturnModel

        if inspect.isclass(ret_ann) and issubclass(ret_ann, BaseModel):
            return ret_ann

        # Special handling for raw dict returns
        origin = get_origin(ret_ann)
        if ret_ann in (dict, Mapping[str, Any]) or origin in (dict, ABCMapping):

            class ReturnDictModel(RootModel[dict]):
                pass

            return ReturnDictModel

        # Special handling for simple types (bool, int, str, etc.)
        # These should be returned as-is, not wrapped in a result field
        if ret_ann in (bool, int, float, str, list, tuple, set):

            class SimpleReturnModel(RootModel[ret_ann]):  # type: ignore
                pass

            return SimpleReturnModel

        # Default case: wrap into a result field for complex types
        return create_model("WrappedReturnModel", result=(ret_ann, ...))

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        parsed_args = self._args_type(**args)
        if not self._is_async:
            import asyncio
            import functools

            loop = asyncio.get_event_loop()
            func_call = functools.partial(self.func, **parsed_args.model_dump())
            return_value = await loop.run_in_executor(None, func_call)
        else:
            return_value = await self.func(**parsed_args.model_dump())

        if isinstance(return_value, BaseModel):
            return return_value.model_dump()

        try:
            if return_value is None:
                parsed_ret = self._return_type()
            else:
                # For RootModel types, we need to handle them differently
                if hasattr(self._return_type, "model_validate"):
                    parsed_ret = self._return_type.model_validate(return_value)
                else:
                    parsed_ret = self._return_type().parse_obj(return_value)
            return parsed_ret.model_dump()
        except ValidationError as e:
            logger.warning(
                f"Return value validation failed for tool '{self._name}': {e}"
            )
            return return_value

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        if self._is_async:
            raise RuntimeError("This tool is async only; please use run_json_async()")
        parsed_args = self._args_type(**args)
        return_value = self.func(**parsed_args.model_dump())

        if isinstance(return_value, BaseModel):
            return return_value.model_dump()

        try:
            # For RootModel types, we need to handle them differently
            if hasattr(self._return_type, "model_validate"):
                parsed_ret = self._return_type.model_validate(return_value)
            else:
                parsed_ret = self._return_type().parse_obj(return_value)
            return parsed_ret.model_dump()
        except ValidationError as e:
            logger.warning(
                f"Return value validation failed for tool '{self._name}': {e}"
            )
            return return_value

    async def save_state_json(self) -> Mapping[str, Any]:
        # Stateless
        return {}

    async def load_state_json(self, state: Mapping[str, Any]) -> None:
        # Stateless
        pass
