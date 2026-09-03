from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from xagent.core.memory.base import MemoryStore
from xagent.core.memory.core import MemoryNote

from ..auth_dependencies import get_current_user
from ..dynamic_memory_store import get_memory_store_manager
from ..models.user import User
from ..user_isolated_memory import UserContext

logger = logging.getLogger(__name__)
MEMORY_READ_UNAVAILABLE_DETAIL = "Memory storage is temporarily unavailable."


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
                                  If not provided, uses the static memory_store.
        """
        if memory_store_provider is not None:
            self.get_memory_store = memory_store_provider
            self._static_memory_store = None
        else:
            # Backward compatibility: use static memory store
            self._static_memory_store = None

        self.router = APIRouter(prefix="/api/memory", tags=["memory"])
        self._setup_routes()

    @property
    def memory_store(self) -> MemoryStore:
        """Get the current memory store (supports dynamic switching)"""
        if (
            hasattr(self, "_static_memory_store")
            and self._static_memory_store is not None
        ):
            return self._static_memory_store
        else:
            return self.get_memory_store()

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
                        if search:
                            memories = self.memory_store.search(
                                query=search,
                                k=1000,
                                filters=filters if filters else None,
                                similarity_threshold=similarity_threshold,
                            )
                        else:
                            memories = self.memory_store.list_all(filters)
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
                with UserContext(int(user.id)):
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
                with UserContext(int(user.id)):
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
                with UserContext(int(user.id)):
                    stats = self.memory_store.get_stats()
                    return MemoryStatsResponse(**stats)
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
            try:
                # Set user context for memory operations
                with UserContext(int(user.id)):
                    # Create new memory note
                    memory_note = MemoryNote(
                        content=memory_request.get("content", ""),
                        keywords=memory_request.get("keywords", []),
                        tags=memory_request.get("tags", []),
                        category=memory_request.get("category", "general"),
                        metadata=memory_request.get("metadata", {}),
                    )

                    response = self.memory_store.add(memory_note)
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

            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Failed to create memory: {str(e)}"
                )

        @self.router.get("/{memory_id}")
        async def get_memory(
            memory_id: str, user: User = Depends(get_current_user)
        ) -> dict[str, Any]:
            try:
                # Set user context for memory operations
                with UserContext(int(user.id)):
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

        @self.router.get("/store-info")
        async def get_store_info(user: User = Depends(get_current_user)) -> dict:
            """Get current memory store information for debugging"""
            try:
                manager = get_memory_store_manager()
                return manager.get_store_info()
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Failed to get store info: {str(e)}"
                )

    def get_router(self) -> APIRouter:
        return self.router
