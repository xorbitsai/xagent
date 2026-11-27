"""Frozen PostgreSQL helper shared by the preview index revision and fresh DDL."""

CREATE_PREVIEW_CONFIG_TEXT = """
CREATE OR REPLACE FUNCTION xagent_preview_config_text_v1(config json, field text)
RETURNS text
LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL UNSAFE
SET search_path = pg_catalog
AS $function$
BEGIN
    RETURN (config -> field)::text;
EXCEPTION
    -- PostgreSQL json accepts these escapes on input but can reject them
    -- while traversing the document, even in an unrelated field.
    WHEN SQLSTATE '22P05' OR SQLSTATE '22021' OR SQLSTATE '22P02' THEN
        RETURN NULL;
END;
$function$
"""

PREVIEW_LOOKUP_KEYS = ("preview_agent_id", "preview_history_agent_id")
PREVIEW_LOOKUP_PREDICATE = (
    "is_visible IS false AND "
    "xagent_preview_config_text_v1(agent_config, 'is_preview') IN ('true', '1')"
)


def preview_lookup_expression(key: str) -> str:
    return f"md5(xagent_preview_config_text_v1(agent_config, '{key}'))"
