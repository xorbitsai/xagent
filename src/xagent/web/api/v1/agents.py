"""SDK management endpoints for user-owned agents."""

from fastapi import APIRouter, Depends, Path, Query, Request

from ...schemas.v1 import (
    AgentRuntimeKeyItem,
    AgentRuntimeKeyRevokeResponse,
    RuntimeKeyResponse,
    V1AgentCreateRequest,
    V1AgentCreateResponse,
    V1AgentResponse,
    V1AgentSkillsResponse,
    V1AgentSkillsUpdateRequest,
    V1AgentSkillsUpdateResponse,
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
    InvalidAgentSkillNamesError,
    InvalidKnowledgeBaseError,
    RuntimeKeySnapshot,
    TemplateNotFoundError,
    WorkforceTemplateNotSupportedError,
)
from ...services.api_keys import (
    AgentKeyMetadata,
    AgentKeyRevocation,
    AgentRuntimeKeyNotFoundError,
    KeyRotationConflict,
)
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


def _runtime_key_item(key: AgentKeyMetadata) -> AgentRuntimeKeyItem:
    return AgentRuntimeKeyItem(
        id=key.id,
        label=key.label,
        key_prefix=key.key_prefix,
        masked_key=key.masked_key,
        status=key.status,
        created_at=key.created_at,
        last_used_at=key.last_used_at,
        revoked_at=key.revoked_at,
    )


def _runtime_key_revoke_response(
    revocation: AgentKeyRevocation,
) -> AgentRuntimeKeyRevokeResponse:
    return AgentRuntimeKeyRevokeResponse(
        id=revocation.key_id,
        key_prefix=revocation.key_prefix,
        revoked=revocation.revoked,
        revoked_at=revocation.revoked_at,
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
                prompt_context_fields=[
                    declaration.model_dump()
                    for declaration in request.prompt_context_fields
                ],
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
    except WorkforceTemplateNotSupportedError:
        raise V1ApiError(
            V1ErrorCode.INVALID_INPUT,
            400,
            "This template creates a workforce, not a single agent, and "
            "cannot be used with this endpoint.",
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


@router.get("/{agent_id}/api-keys", response_model=list[AgentRuntimeKeyItem])
async def list_agent_runtime_keys(
    # ``agents.id`` is a 32-bit integer column.
    agent_id: int = Path(gt=0, le=2**31 - 1),
    include_revoked: bool = Query(False),
    authed: tuple[UserPrincipalSnapshot, PersonalApiKeySnapshot] = Depends(
        get_user_from_personal_key
    ),
) -> list[AgentRuntimeKeyItem]:
    """List the runtime keys of an agent the key holder can manage.

    Revoked keys are left out unless ``include_revoked`` is set. No response
    carries a key's secret.
    """
    user, _key = authed
    keys = await AgentManagementRuntime().list_agent_runtime_keys(
        user_id=int(user.id), agent_id=agent_id, include_revoked=include_revoked
    )
    if keys is None:
        raise V1ApiError(V1ErrorCode.AGENT_NOT_FOUND, 404)
    return [_runtime_key_item(key) for key in keys]


@router.delete(
    "/{agent_id}/api-keys/{key_id}", response_model=AgentRuntimeKeyRevokeResponse
)
async def revoke_agent_runtime_key(
    # ``agents.id`` and ``agent_api_keys.id`` are 32-bit integer columns.
    agent_id: int = Path(gt=0, le=2**31 - 1),
    key_id: int = Path(gt=0, le=2**31 - 1),
    authed: tuple[UserPrincipalSnapshot, PersonalApiKeySnapshot] = Depends(
        get_user_from_personal_key
    ),
) -> AgentRuntimeKeyRevokeResponse:
    """Revoke one runtime key of an agent the key holder can manage.

    The agent's other keys keep working. Revoking an already revoked key is
    a no-op that reports ``revoked=false``. A key id that is not bound to
    this agent is ``api_key_not_found`` whether or not it exists elsewhere.
    """
    user, key = authed
    try:
        revocation = await AgentManagementRuntime().revoke_agent_runtime_key(
            user_id=int(user.id),
            credential_ref=f"personal_key:{key.key_prefix}",
            agent_id=agent_id,
            key_id=key_id,
        )
    except AgentRuntimeKeyNotFoundError:
        raise V1ApiError(V1ErrorCode.API_KEY_NOT_FOUND, 404)
    if revocation is None:
        raise V1ApiError(V1ErrorCode.AGENT_NOT_FOUND, 404)
    return _runtime_key_revoke_response(revocation)


@router.get("/{agent_id}/skills", response_model=V1AgentSkillsResponse)
async def get_agent_skills(
    # ``agents.id`` is a 32-bit integer column.
    agent_id: int = Path(gt=0, le=2**31 - 1),
    authed: tuple[UserPrincipalSnapshot, PersonalApiKeySnapshot] = Depends(
        get_user_from_personal_key
    ),
) -> V1AgentSkillsResponse:
    """Read the skill names of an agent the key holder can see."""
    user, _key = authed
    skills = await AgentManagementRuntime().get_agent_skills(
        user_id=int(user.id), agent_id=agent_id
    )
    if skills is None:
        raise V1ApiError(V1ErrorCode.AGENT_NOT_FOUND, 404)
    return V1AgentSkillsResponse(agent_id=agent_id, skills=list(skills))


@router.put("/{agent_id}/skills", response_model=V1AgentSkillsUpdateResponse)
async def set_agent_skills(
    request: V1AgentSkillsUpdateRequest,
    # ``agents.id`` is a 32-bit integer column.
    agent_id: int = Path(gt=0, le=2**31 - 1),
    authed: tuple[UserPrincipalSnapshot, PersonalApiKeySnapshot] = Depends(
        get_user_from_personal_key
    ),
) -> V1AgentSkillsUpdateResponse:
    """Replace the skill names of an agent the key holder can see."""
    user, key = authed
    try:
        result = await AgentManagementRuntime().set_agent_skills(
            user_id=int(user.id),
            key_prefix=key.key_prefix,
            agent_id=agent_id,
            skills=request.skills,
        )
    except InvalidAgentSkillNamesError as e:
        raise V1ApiError(V1ErrorCode.INVALID_INPUT, 400, str(e))
    if result is None:
        raise V1ApiError(V1ErrorCode.AGENT_NOT_FOUND, 404)
    return V1AgentSkillsUpdateResponse(
        agent_id=agent_id,
        skills=list(result.skills),
        unresolved_for_caller=list(result.unresolved_for_caller),
    )
