from xagent.web.services import agent_team_scope as agent_scope
from xagent.web.services import connector_team_scope as connector_scope
from xagent.web.services import knowledge_base_team_scope as kb_scope


def test_agent_team_hooks_install_as_one_group():
    scope = agent_scope.AgentTeamScope(team_id=42, is_team_admin=True)
    agent_scope.set_agent_team_hooks(
        scope=lambda db, user_id: scope,
        connector_validator=lambda db, user_id, team_id, tools: [{"id": 1}],
        knowledge_base_validator=lambda db, user_id, team_id, names: [{"name": "kb"}],
    )
    try:
        assert agent_scope.get_agent_team_scope(None, 7) == scope
        assert agent_scope.validate_team_agent_connectors(None, 7, 42, []) == [
            {"id": 1}
        ]
        assert agent_scope.validate_team_agent_knowledge_bases(None, 7, 42, []) == [
            {"name": "kb"}
        ]
    finally:
        agent_scope.set_agent_team_hooks()


def test_connector_team_hooks_delegate_and_reset():
    deleted_calls = []
    renamed_calls = []
    access_calls = []

    with connector_scope.snapshot_connector_team_hooks():
        connector_scope.set_connector_team_hooks(
            visibility=lambda db, user_id: {"mcp": {11}, "custom_api": {22}},
            team_visibility=lambda db, *, team_id: {"mcp": {33}, "custom_api": {44}},
            deleted=lambda db, user_id, kind, connector_id: (
                deleted_calls.append((db, user_id, kind, connector_id))
                or connector_scope.ConnectorDeleteDecision(
                    team_owned=True, authorized=True, delete_definition=False
                )
            ),
            renamed=lambda db, user_id, kind, cid, old, new: renamed_calls.append(
                (db, user_id, kind, cid, old, new)
            ),
            access=lambda db, user_id, refs: (
                access_calls.append((db, user_id, refs))
                or {
                    ref: connector_scope.ConnectorAccess(team_owned=True, can_edit=True)
                    for ref in refs
                }
            ),
        )
        assert connector_scope.visible_team_connector_ids(None, 7) == {
            "mcp": {11},
            "custom_api": {22},
        }
        assert connector_scope.team_connector_ids(None, team_id=9) == {
            "mcp": {33},
            "custom_api": {44},
        }
        decision = connector_scope.delete_team_connector(None, 7, "mcp", 11)
        assert decision.team_owned and decision.authorized
        connector_scope.rename_team_connector(None, 7, "mcp", 11, "old", "new")
        access = connector_scope.resolve_connector_access(None, 7, [("mcp", 11)])
        assert access == {
            ("mcp", 11): connector_scope.ConnectorAccess(team_owned=True, can_edit=True)
        }
        assert deleted_calls == [(None, 7, "mcp", 11)]
        assert renamed_calls == [(None, 7, "mcp", 11, "old", "new")]
        assert access_calls == [(None, 7, frozenset({("mcp", 11)}))]

        # A reset-all call clears every slot, including the access one --
        # this is the property the snapshot primitive above restores after
        # this block exits, so the reset stays inside the block rather than
        # standing in for one.
        connector_scope.set_connector_team_hooks()
        assert connector_scope.team_connector_hook_installed() is False
        assert connector_scope.resolve_connector_access(None, 7, [("mcp", 11)]) == {}


def test_knowledge_base_team_hooks_delegate_with_none_session():
    lifecycle_calls = []
    access = kb_scope.KnowledgeBaseAccess(
        name="shared", storage_user_id=42, team_owned=True
    )
    team_access = kb_scope.KnowledgeBaseAccess(
        name="team-doc",
        storage_user_id=99,
        team_owned=True,
        can_edit=False,
        can_delete=False,
    )
    with kb_scope.snapshot_knowledge_base_team_hooks():
        kb_scope.set_knowledge_base_team_hooks(
            visibility=lambda db, user_id: [access],
            access=lambda db, user_id, name, action: access,
            renamed=lambda db, user_id, old, new: lifecycle_calls.append(
                ("rename", db, user_id, old, new)
            ),
            deleted=lambda db, user_id, name, new: lifecycle_calls.append(
                ("delete", db, user_id, name, new)
            ),
            team_visibility=lambda db, *, team_id: [team_access],
        )
        assert kb_scope.visible_team_knowledge_bases(None, 7) == [access]
        assert kb_scope.resolve_knowledge_base_access(None, 7, "shared") == access
        kb_scope.notify_knowledge_base_renamed(None, 42, "old", "new")
        kb_scope.notify_knowledge_base_deleted(None, 42, "new")
        assert lifecycle_calls == [
            ("rename", None, 42, "old", "new"),
            ("delete", None, 42, "new", None),
        ]
        assert kb_scope.team_knowledge_bases(None, team_id=5) == [team_access]
        assert kb_scope.team_knowledge_base_hook_installed() is True

        # A reset-all call clears every slot, including the team-keyed one --
        # this is the property the snapshot primitive above restores after
        # this block exits.
        kb_scope.set_knowledge_base_team_hooks()
        assert kb_scope.team_knowledge_base_hook_installed() is False
        assert kb_scope.visible_team_knowledge_bases(None, 7) == []
