"""Connector runtime requirements: the shared response shape both read
endpoints return.

The agent-keyed and task-keyed read endpoints both return this same
requirements report -- which runtime inputs a task's (or a prospective
task's) connectors declare, and whether each one already has a value --
never a stored value itself, and never a connector's transport or
authentication configuration -- the same shape from both, with one field,
``satisfied``, answering a different question on each; see
``ConnectorRuntimeRequirementsModel``.

Placed in its own module rather than ``schemas/chat.py`` because a values-
submission endpoint lands on top of it shortly and will share this same
response shape as its own 200 body; putting it here now avoids a later
move that would touch every existing importer.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ConnectorRuntimeRefModel(BaseModel):
    """Wire identity of a connector, as returned in a requirements report."""

    connector_type: str
    connector_id: int


class ConnectorRuntimeInputModel(BaseModel):
    """One declared runtime input and whether it is currently satisfied.

    ``key`` is the raw key name a connector owner wrote when declaring the
    input -- there is no human-readable label anywhere in the declaration.
    ``type`` is already normalized server-side to ``"string"`` or
    ``"object"``; a client must not normalize it again or expect any other
    value. ``expired`` is a constant ``False`` in every section at this
    phase; a later phase that adds a real secret store gives it a real
    value without changing its meaning. ``satisfied`` is likewise a
    constant ``False`` for the ``secrets`` and ``auth_selector`` sections,
    because no secret store exists yet to hold such a value, and it is the
    ``section`` field that tells the two kinds of ``False`` apart: in the
    ``context`` section ``False`` means the value has not been supplied yet
    and can be, while in ``secrets`` and ``auth_selector`` it means no
    value can be supplied at this phase at all. One ``context`` key is
    also always ``False``: a key whose name the per-turn gate rejects as
    malformed, which is reported so that no key of that kind, required or
    not, can let the report read as met.

    ``section`` and ``type`` are closed sets, and a client may switch on
    them exhaustively: the server emits no other value in either field,
    and adding one would be a wire change.
    """

    section: Literal["context", "secrets", "auth_selector"]
    key: str
    type: Literal["string", "object"]
    required: bool
    satisfied: bool
    expired: bool = False


class ConnectorRuntimeConnectorModel(BaseModel):
    """One connector's declared runtime inputs.

    ``name`` is the only piece of connector identity beyond the ref that is
    ever included -- never the connector's URL, headers, environment, or
    authentication configuration.
    """

    connector_ref: ConnectorRuntimeRefModel
    name: str
    inputs: list[ConnectorRuntimeInputModel]


class ConnectorRuntimeRequirementsModel(BaseModel):
    """A requirements report. Every field always appears.

    ``connectors`` is empty, never omitted, when nothing is selected or
    declares a runtime input. ``secrets_expires_at`` is a constant ``null``
    in this phase; a later phase gives it a real value without changing
    its meaning or making it optional.

    ``satisfied`` answers a different question on each endpoint that
    returns this model, and a client must read it against the endpoint it
    called. The agent-keyed report has no task, so no value can be stored
    against it: there ``satisfied`` answers "a task created from this agent
    right now would need no further input", i.e. nothing required is
    declared at all. The task-create response answers that same question:
    it is computed from the agent, before the new task is persisted and so
    before any value could have been stored against it. Only the
    task-keyed report consults stored values, and only there does
    ``satisfied`` answer "every required input of this task already has a
    value". The per-input ``satisfied`` follows the same split: always
    ``False`` on the agent-keyed report and on the task-create response,
    and a real per-key answer on the task-keyed report.

    ``satisfied`` is also ``False``, on every endpoint and whatever the
    per-input flags say, while any listed key's name is one the per-turn
    gate rejects as malformed -- required or not, because that gate
    refuses the whole turn over such a key rather than only over a
    required one.
    """

    satisfied: bool
    secrets_expires_at: str | None
    connectors: list[ConnectorRuntimeConnectorModel]
