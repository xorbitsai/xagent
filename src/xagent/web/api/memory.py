from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Iterator, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from xagent.core.memory.base import MemoryStore
from xagent.core.memory.core import MemoryNote

from ..auth_dependencies import get_current_user
from ..dynamic_memory_store import get_memory_store, get_memory_store_manager
from ..memory_lifecycle import MemoryUnavailableError
from ..models.user import User
from ..services.db_runtime import run_db_io_cancellation_safe
from ..user_isolated_memory import UserContext

logger = logging.getLogger(__name__)
MEMORY_READ_UNAVAILABLE_DETAIL = "Memory storage is temporarily unavailable."


@contextmanager
def memory_operation() -> Iterator[None]:
    """Answer a revocation that lands mid-operation with the same stable 503.

    :attr:`MemoryManagementRouter.memory_store` maps only the *initial*
    acquisition failure. The store it returns is a revocable proxy that
    re-checks its publication generation on every call, so a concurrent
    revalidation -- an administrator changing the authority's vector space,
    another worker revoking the publication -- can raise
    :class:`MemoryUnavailableError` from the delegated operation instead,
    after the acquisition already succeeded.

    Without this, each route's broad ``except Exception`` turned that into a
    500 naming the internal failure. One shared boundary keeps every route on
    the single public-safe detail, and keeps the seven of them from drifting
    apart. ``store-info`` deliberately does not use it: it reports the
    lifecycle rather than operating on the store, and answers 200 throughout.
    """
    try:
        yield
    except MemoryUnavailableError as error:
        raise HTTPException(status_code=503, detail=error.status.detail) from None


class MemoryListRequest(BaseModel):
    category: Optional[str] = Field(None, description="Filter by memory category")
    tags: Optional[list[str]] = Field(
        None, description="Filter by tags (all must match)"
    )
    keywords: Optional[list[str]] = Field(
        None, description="Filter by keywords (all must match)"
    )
    date_from: Optional[datetime] = Field(
        None, description="Filter memories from this date"
    )
    date_to: Optional[datetime] = Field(
        None, description="Filter memories to this date"
    )
    limit: Optional[int] = Field(
        100, description="Maximum number of memories to return"
    )
    offset: Optional[int] = Field(0, description="Offset for pagination")


class MemoryUpdateRequest(BaseModel):
    content: Optional[str] = Field(None, description="Updated memory content")
    keywords: Optional[list[str]] = Field(None, description="Updated memory keywords")
    tags: Optional[list[str]] = Field(None, description="Updated memory tags")
    category: Optional[str] = Field(None, description="Updated memory category")
    metadata: Optional[dict[str, Any]] = Field(
        None, description="Updated memory metadata"
    )


class MemoryListResponse(BaseModel):
    memories: list[dict[str, Any]]
    total_count: int
    filters_used: dict[str, Any]


class MemoryStatsResponse(BaseModel):
    total_count: int
    category_counts: dict[str, int]
    tag_counts: dict[str, int]
    memory_store_type: str
    error: Optional[str] = None


class MemoryManagementRouter:
    def __init__(
        self, memory_store_provider: Optional[Callable[[], MemoryStore]] = None
    ) -> None:
        """
        Initialize memory management router.

        Args:
            memory_store_provider: Optional function that returns a memory store.
                                  If not provided, uses the admitted store
                                  published by the lifecycle manager.
        """
        self.get_memory_store = (
            memory_store_provider
            if memory_store_provider is not None
            else get_memory_store
        )

        self.router = APIRouter(prefix="/api/memory", tags=["memory"])
        self._setup_routes()

    @property
    def memory_store(self) -> MemoryStore:
        """Get the published memory store, or fail closed with a 503.

        Every lifecycle condition the runtime can be in -- awaiting a retryable
        admission, blocked on repair, missing a usable credential, or needing a
        restart after meaningful drift -- reaches callers as the same stable,
        public-safe 503. Only the operator log carries the detail.
        """
        try:
            return self.get_memory_store()
        except MemoryUnavailableError as error:
            raise HTTPException(status_code=503, detail=error.status.detail) from None

    def _setup_routes(self) -> None:
        @self.router.get("/list", response_model=MemoryListResponse)
        async def list_memories(
            category: Optional[str] = Query(None, description="Filter by category"),
            tags: Optional[str] = Query(
                None, description="Comma-separated tags to filter"
            ),
            keywords: Optional[str] = Query(
                None, description="Comma-separated keywords to filter"
            ),
            date_from: Optional[datetime] = Query(
                None, description="Filter from this date"
            ),
            date_to: Optional[datetime] = Query(
                None, description="Filter to this date"
            ),
            search: Optional[str] = Query(
                None, description="Search query to filter memories by content"
            ),
            similarity_threshold: Optional[float] = Query(
                None, description="Similarity threshold for vector search (0.1-2.0)"
            ),
            limit: int = Query(
                100, ge=1, le=1000, description="Maximum results to return"
            ),
            offset: int = Query(0, ge=0, description="Offset for pagination"),
            user: User = Depends(get_current_user),
        ) -> MemoryListResponse:
            try:
                # Set user context for memory operations
                with UserContext(int(user.id)):
                    # Build filters
                    filters: dict[str, Any] = {}
                    if category:
                        filters["category"] = category
                    if tags:
                        filters["tags"] = [
                            tag.strip() for tag in tags.split(",") if tag.strip()
                        ]
                    if keywords:
                        filters["keywords"] = [
                            kw.strip() for kw in keywords.split(",") if kw.strip()
                        ]
                    if date_from:
                        filters["date_from"] = date_from
                    if date_to:
                        filters["date_to"] = date_to

                    try:
                        with memory_operation():
                            if search:
                                memories = self.memory_store.search(
                                    query=search,
                                    k=1000,
                                    filters=filters if filters else None,
                                    similarity_threshold=similarity_threshold,
                                )
                            else:
                                memories = self.memory_store.list_all(filters)
                    except HTTPException:
                        raise
                    except Exception:
                        logger.exception("Memory list read failed")
                        raise HTTPException(
                            status_code=503,
                            detail=MEMORY_READ_UNAVAILABLE_DETAIL,
                        ) from None

                    # Apply pagination
                    total_count = len(memories)
                    memories = memories[offset : offset + limit]

                    # Convert to dict format for response
                    memory_dicts = []
                    for memory in memories:
                        memory_dict = {
                            "id": memory.id,
                            "content": memory.content,
                            "keywords": memory.keywords,
                            "tags": memory.tags,
                            "category": memory.category,
                            "timestamp": memory.timestamp,
                            "mime_type": memory.mime_type,
                            "metadata": memory.metadata,
                        }
                        memory_dicts.append(memory_dict)

                    return MemoryListResponse(
                        memories=memory_dicts,
                        total_count=total_count,
                        filters_used=filters,
                    )
            except HTTPException:
                raise
            except Exception:
                logger.exception("Failed to build memory list response")
                raise HTTPException(
                    status_code=500, detail="Failed to list memories."
                ) from None

        @self.router.delete("/{memory_id}")
        async def delete_memory(
            memory_id: str, user: User = Depends(get_current_user)
        ) -> dict[str, Any]:
            try:
                # Set user context for memory operations
                with UserContext(int(user.id)), memory_operation():
                    response = self.memory_store.delete(memory_id)
                    if response.success:
                        return {
                            "success": True,
                            "message": "Memory deleted successfully",
                        }
                    else:
                        raise HTTPException(
                            status_code=404, detail=response.error or "Memory not found"
                        )
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Failed to delete memory: {str(e)}"
                )

        @self.router.put("/{memory_id}")
        async def update_memory(
            memory_id: str,
            update_request: MemoryUpdateRequest,
            user: User = Depends(get_current_user),
        ) -> dict[str, Any]:
            try:
                # Set user context for memory operations
                with UserContext(int(user.id)), memory_operation():
                    # Get existing memory
                    get_response = self.memory_store.get(memory_id)
                    if not get_response.success:
                        raise HTTPException(status_code=404, detail="Memory not found")

                    existing_memory = get_response.content
                    if not isinstance(existing_memory, MemoryNote):
                        raise HTTPException(
                            status_code=500, detail="Invalid memory data"
                        )

                    # Update fields that are provided
                    updates: dict[str, Any] = {}
                    if update_request.content is not None:
                        updates["content"] = update_request.content
                    if update_request.keywords is not None:
                        updates["keywords"] = update_request.keywords
                    if update_request.tags is not None:
                        updates["tags"] = update_request.tags
                    if update_request.category is not None:
                        updates["category"] = update_request.category
                    if update_request.metadata is not None:
                        updates["metadata"] = update_request.metadata

                    # Create updated memory note
                    updated_memory = MemoryNote(
                        id=memory_id,
                        content=updates.get("content", existing_memory.content),
                        keywords=updates.get("keywords", existing_memory.keywords),
                        tags=updates.get("tags", existing_memory.tags),
                        category=updates.get("category", existing_memory.category),
                        metadata=updates.get("metadata", existing_memory.metadata),
                        mime_type=existing_memory.mime_type,
                        timestamp=existing_memory.timestamp,  # Keep original timestamp
                    )

                    # Update in store
                    response = self.memory_store.update(updated_memory)
                    if response.success:
                        return {
                            "success": True,
                            "message": "Memory updated successfully",
                        }
                    else:
                        raise HTTPException(
                            status_code=500,
                            detail=response.error or "Failed to update memory",
                        )

            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Failed to update memory: {str(e)}"
                )

        @self.router.get("/stats", response_model=MemoryStatsResponse)
        async def get_memory_stats(
            user: User = Depends(get_current_user),
        ) -> MemoryStatsResponse:
            try:
                # Set user context for memory operations
                with UserContext(int(user.id)), memory_operation():
                    stats = self.memory_store.get_stats()
                    return MemoryStatsResponse(**stats)
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Failed to get memory stats: {str(e)}"
                )

        @self.router.post("")
        async def create_memory(
            memory_request: dict[str, Any], user: User = Depends(get_current_user)
        ) -> dict[str, Any]:
            # Validate required fields
            if "content" not in memory_request:
                raise HTTPException(status_code=422, detail="Content field is required")
            # Resolved before the try block on purpose: the lifecycle 503 must
            # reach the caller as itself, while the handler below keeps
            # wrapping store failures the way it always has.
            store = self.memory_store
            try:
                # Set user context for memory operations
                with UserContext(int(user.id)), memory_operation():
                    # Create new memory note
                    memory_note = MemoryNote(
                        content=memory_request.get("content", ""),
                        keywords=memory_request.get("keywords", []),
                        tags=memory_request.get("tags", []),
                        category=memory_request.get("category", "general"),
                        metadata=memory_request.get("metadata", {}),
                    )

                    response = store.add(memory_note)
                    if response.success:
                        return {
                            "success": True,
                            "memory_id": response.memory_id,
                            "message": "Memory created successfully",
                        }
                    else:
                        raise HTTPException(
                            status_code=500,
                            detail=response.error or "Failed to create memory",
                        )

            except HTTPException:
                # Needed for the 503 above to reach the caller at all: without
                # it the broad handler re-wrapped every HTTPException raised
                # inside the block, status code included, as a 500.
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Failed to create memory: {str(e)}"
                )

        # Registered before the "/{memory_id}" catch-all, which would otherwise
        # swallow this path and answer it as a lookup for a note called
        # "store-info".
        @self.router.get("/store-info")
        async def get_store_info(user: User = Depends(get_current_user)) -> dict:
            """Report the memory lifecycle state.

            This stays available in every lifecycle state, including
            BLOCKED_REPAIR, so an operator can see why memory is fenced off
            while the rest of the deployment keeps serving.
            """
            try:
                # Off the event loop, like every other caller of the manager:
                # the report reaches the drift check, which checks out a
                # synchronous authority Session. Running that inline would
                # block the loop thread while ``get_current_user``'s
                # request-scoped connection is still held, so on a
                # single-slot pool the nested checkout would be waiting on
                # the very thread that has to release it.
                return await run_db_io_cancellation_safe(
                    get_memory_store_manager().get_store_info
                )
            except Exception:
                logger.exception("Failed to read memory store info")
                raise HTTPException(
                    status_code=500, detail="Failed to get store info."
                ) from None

        @self.router.get("/{memory_id}")
        async def get_memory(
            memory_id: str, user: User = Depends(get_current_user)
        ) -> dict[str, Any]:
            try:
                # Set user context for memory operations
                with UserContext(int(user.id)), memory_operation():
                    response = self.memory_store.get(memory_id)
                    if response.success and response.content:
                        memory = response.content
                        if isinstance(memory, MemoryNote):
                            return {
                                "id": memory.id,
                                "content": memory.content,
                                "keywords": memory.keywords,
                                "tags": memory.tags,
                                "category": memory.category,
                                "timestamp": memory.timestamp,
                                "mime_type": memory.mime_type,
                                "metadata": memory.metadata,
                            }
                        else:
                            return {"content": memory}
                    else:
                        raise HTTPException(
                            status_code=404, detail=response.error or "Memory not found"
                        )
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Failed to get memory: {str(e)}"
                )

    def get_router(self) -> APIRouter:
        return self.router
