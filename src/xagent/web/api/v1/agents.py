"""SDK management endpoints for user-owned agents."""

from fastapi import APIRouter, Depends, Request

from ...schemas.v1 import (
    RuntimeKeyResponse,
    V1AgentCreateRequest,
    V1AgentCreateResponse,
    V1AgentResponse,
    V1AgentSummary,
    V1AgentTemplateCreateRequest,
)
from ...services.agent_management import (
    AgentCreateSpec,
    AgentManagementRuntime,
    AgentResponseSnapshot,
    AgentSummarySnapshot,
    DuplicateAgentNameError,
    InvalidAgentModelConfigError,
    InvalidKnowledgeBaseError,
    RuntimeKeySnapshot,
    TemplateNotFoundError,
)
from ...services.api_keys import KeyRotationConflict
from .deps import (
    PersonalApiKeySnapshot,
    UserPrincipalSnapshot,
    get_user_from_personal_key,
)
from .errors import V1ApiError, V1ErrorCode

router = APIRouter(prefix="/agents")


def _runtime_key_response(api_key: RuntimeKeySnapshot) -> RuntimeKeyResponse:
    return RuntimeKeyResponse(
        full_key=api_key.full_key,
        key_prefix=api_key.key_prefix,
        created_at=api_key.created_at,
    )


def _agent_response(agent: AgentResponseSnapshot) -> V1AgentResponse:
    return V1AgentResponse.model_validate(agent.to_response_dict())


def _agent_summary(agent: AgentSummarySnapshot) -> V1AgentSummary:
    return V1AgentSummary(
        id=agent.id,
        name=agent.name,
        description=agent.description,
        logo_url=agent.logo_url,
        status=agent.status,
        created_at=agent.created_at,
        updated_at=agent.updated_at,
        widget_enabled=agent.widget_enabled,
        allowed_domains=list(agent.allowed_domains),
        share_enabled=agent.share_enabled,
        share_updated_at=agent.share_updated_at,
    )


@router.get("", response_model=list[V1AgentSummary])
async def list_agents(
    authed: tuple[UserPrincipalSnapshot, PersonalApiKeySnapshot] = Depends(
        get_user_from_personal_key
    ),
) -> list[V1AgentSummary]:
    user, _key = authed
    agents = await AgentManagementRuntime().list_agents(user_id=int(user.id))
    return [_agent_summary(agent) for agent in agents]


@router.post("", response_model=V1AgentCreateResponse)
async def create_agent(
    request: V1AgentCreateRequest,
    authed: tuple[UserPrincipalSnapshot, PersonalApiKeySnapshot] = Depends(
        get_user_from_personal_key
    ),
) -> V1AgentCreateResponse:
    user, _key = authed
    try:
        result = await AgentManagementRuntime().create_agent(
            user_id=int(user.id),
            is_admin=bool(user.is_admin),
            spec=AgentCreateSpec.from_values(
                name=request.name,
                description=request.description,
                instructions=request.instructions,
                execution_mode=request.execution_mode,
                models=request.models,
                knowledge_bases=request.knowledge_bases,
                skills=request.skills,
                tool_categories=request.tool_categories,
                suggested_prompts=request.suggested_prompts,
                generate_runtime_key=request.generate_runtime_key,
            ),
        )
    except DuplicateAgentNameError:
        raise V1ApiError(
            V1ErrorCode.INVALID_INPUT, 400, "Agent with this name already exists."
        )
    except InvalidAgentModelConfigError:
        raise V1ApiError(
            V1ErrorCode.INVALID_INPUT,
            400,
            "Agent models must use integer DB model ids for known model slots.",
        )
    except InvalidKnowledgeBaseError as e:
        raise V1ApiError(V1ErrorCode.INVALID_INPUT, 400, str(e))
    except KeyRotationConflict:
        raise V1ApiError(
            V1ErrorCode.INTERNAL_ERROR, 409, "Runtime key rotation conflict."
        )

    return V1AgentCreateResponse(
        agent=_agent_response(result.agent),
        api_key=(
            _runtime_key_response(result.api_key)
            if result.api_key is not None
            else None
        ),
    )


@router.post("/from-template", response_model=V1AgentCreateResponse)
async def create_agent_from_template(
    request: V1AgentTemplateCreateRequest,
    fastapi_request: Request,
    authed: tuple[UserPrincipalSnapshot, PersonalApiKeySnapshot] = Depends(
        get_user_from_personal_key
    ),
) -> V1AgentCreateResponse:
    user, _key = authed
    template_manager = getattr(fastapi_request.app.state, "template_manager", None)
    try:
        result = await AgentManagementRuntime(
            template_manager=template_manager
        ).create_agent_from_template(
            user_id=int(user.id),
            is_admin=bool(user.is_admin),
            template_id=request.template_id,
            name=request.name,
            description=request.description,
            instructions=request.instructions,
            execution_mode=request.execution_mode,
            models=request.models,
            knowledge_bases=request.knowledge_bases,
            skills=request.skills,
            tool_categories=request.tool_categories,
            suggested_prompts=request.suggested_prompts,
            generate_runtime_key=request.generate_runtime_key,
        )
    except TemplateNotFoundError:
        raise V1ApiError(V1ErrorCode.TEMPLATE_NOT_FOUND, 404)
    except DuplicateAgentNameError:
        raise V1ApiError(
            V1ErrorCode.INVALID_INPUT, 400, "Agent with this name already exists."
        )
    except InvalidAgentModelConfigError:
        raise V1ApiError(
            V1ErrorCode.INVALID_INPUT,
            400,
            "Agent models must use integer DB model ids for known model slots.",
        )
    except InvalidKnowledgeBaseError as e:
        raise V1ApiError(V1ErrorCode.INVALID_INPUT, 400, str(e))
    except KeyRotationConflict:
        raise V1ApiError(
            V1ErrorCode.INTERNAL_ERROR, 409, "Runtime key rotation conflict."
        )

    return V1AgentCreateResponse(
        agent=_agent_response(result.agent),
        api_key=(
            _runtime_key_response(result.api_key)
            if result.api_key is not None
            else None
        ),
    )


@router.post("/{agent_id}/api-key", response_model=RuntimeKeyResponse)
async def rotate_agent_runtime_key(
    agent_id: int,
    authed: tuple[UserPrincipalSnapshot, PersonalApiKeySnapshot] = Depends(
        get_user_from_personal_key
    ),
) -> RuntimeKeyResponse:
    user, _key = authed
    try:
        api_key = await AgentManagementRuntime().rotate_agent_runtime_key(
            user_id=int(user.id), agent_id=agent_id
        )
    except KeyRotationConflict:
        raise V1ApiError(
            V1ErrorCode.INTERNAL_ERROR, 409, "Runtime key rotation conflict."
        )
    if api_key is None:
        raise V1ApiError(V1ErrorCode.AGENT_NOT_FOUND, 404)
    return _runtime_key_response(api_key)
