"""Connector smoke V2: check declarations, identity adapters and run results.

- ``declarations``: the declaration format and its validation.
- ``fixtures``: fixture keys and their resolution.
- ``registry``: each connector's declaration, declared or derived from L3.
- ``identity``: the identity adapter protocol the engine runs through.
- ``results``: the run result, its JSON document and ``result.schema.json``.
- ``assertions``: judging one tool call's result against a check.
- ``engine``: ``run_checks``, which runs an identity's checks into a result.

The format is documented in ``docs/operations/connector-smoke-v2.md``.
"""
