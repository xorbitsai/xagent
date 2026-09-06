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
    malformed, which is reported so a required key of that kind cannot
    let the report read as met.
    """

    section: str
    key: str
    type: str
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
    declared at all. The task-keyed report and the task-create response
    answer "every required input of this task already has a value". The
    per-input ``satisfied`` follows the same split: always ``False`` on the
    agent-keyed report, and a real per-key answer on the other two.
    """

    satisfied: bool
    secrets_expires_at: str | None
    connectors: list[ConnectorRuntimeConnectorModel]
