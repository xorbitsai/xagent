# Skills Directory

This directory contains the skills system for xagent.

## Structure

```
skills/
├── builtin/              # Built-in skills (committed to git)
│   ├── code_reviewer/
│   │   ├── SKILL.md
│   │   └── template.md
│   └── test_generator/
│       ├── SKILL.md
│       └── template.md
└── manager.py           # Skill manager implementation
```

## Built-in Skills

Built-in skills are located in `src/xagent/skills/builtin/` and are committed to the repository.

> **Note:** Currently no built-in skills are included. Add your own skills here to ship with the application.

## User Skills

Users can add custom skills in `.xagent/skills/` (outside the `src/` directory).

User skills:
- Are not committed to git (see `.gitignore`)
- Override built-in skills with the same name
- Can be added without modifying the source code

## Skill Format

Each skill is a directory containing:

- **SKILL.md** (required): Entry point with description, when to use, execution flow
- **template.md** (optional): Prompt template for the skill
- **examples/** (optional): Example files
- **resources/** (optional): Additional resources

See individual skill directories for examples.

### Routing metadata

`description` and `when_to_use` decide whether a skill gets loaded at all: they
are the only thing the model sees about a skill it has not loaded yet. Two rules
govern them, and they apply to every skill — built-in, user, external, or
provider-backed.

**Frontmatter is authoritative; a body section is only a fallback.** A scalar
`description:` / `when_to_use:` in the YAML frontmatter wins over a `##
Description` / `## When to Use` section. A whitespace-only frontmatter value
counts as absent and falls through to the section.

```markdown
---
name: my-skill
description: |
  What this produces, plus the words a user would say when they want it.
when_to_use: |
  When to reach for this, and which skill to prefer instead when not.
---

## Description

Longer prose for after the skill is loaded. Ignored for routing when the
frontmatter above defines the field.
```

**Each field is capped at 200 characters in the index.** Whitespace is
collapsed to single spaces first, and anything past the cap is dropped
silently — trigger vocabulary and "use skill X instead" cross-references are
what usually get lost, since they tend to sit at the end. Keep both fields
under the cap and put the detail in the body, which is injected in full once
the skill is loaded. `tests/skills/test_builtin_index_entries.py` enforces this
for built-in skills.

## Adding a New Built-in Skill

1. Create a new directory in `src/xagent/skills/builtin/your_skill/`
2. Add `SKILL.md` with the skill description
3. Optionally add `template.md` or other files
4. The skill will be automatically loaded on startup

## Adding a User Skill

1. Create a directory in `.xagent/skills/your_skill/`
2. Add `SKILL.md` and any other files
3. Restart the server or call `POST /api/skills/reload`

## Configuration

### Environment Variables

You can add custom skills directories by setting the `XAGENT_EXTERNAL_SKILLS_LIBRARY_DIRS` environment variable. These directories are **appended** to the default built-in and user skills directories:

```bash
# Single directory (appended to defaults)
XAGENT_EXTERNAL_SKILLS_LIBRARY_DIRS="/path/to/custom/skills"

# Multiple directories (comma-separated, all appended)
XAGENT_EXTERNAL_SKILLS_LIBRARY_DIRS="/path/to/skills1,/path/to/skills2,~/skills"

# With path expansion
XAGENT_EXTERNAL_SKILLS_LIBRARY_DIRS="~/skills,$HOME/custom_skills,./local_skills"

# Load order: built-in -> user -> external (later override earlier)
```

**Important**: External directories are always appended to the default directories. The final load order is:
1. Built-in skills (`src/xagent/skills/builtin/`)
2. User skills (`.xagent/skills/`)
3. External directories from `XAGENT_EXTERNAL_SKILLS_LIBRARY_DIRS`

Skills with the same name are loaded in order, with later directories overriding earlier ones.

## Provider extension contract

Applications may register a read provider with `set_skill_library_provider()` or
a write provider with `set_skill_write_provider()`. Both provider protocols and
their registration functions are exported from `xagent.skills`.

Read providers receive `SkillScopeContext`; write providers receive
`SkillWriteContext`. These contexts contain only a `user_id` and copied scalar
metadata. They never contain an ORM entity, request, database session, or
session factory, and should not be serialized as a transport contract. Metadata
is non-authoritative: a provider must resolve current authorization from
`user_id` inside its own short-lived dependency scope.

A database-backed provider owns its complete operation: open a session,
authorize, materialize its result, commit on success or roll back on failure,
and close before returning. It must not commit or roll back the caller's
request transaction.

Expected public write failures must raise `SkillWriteProviderError` with an
allowlisted `SkillWriteProviderErrorReason` and a deliberately public-safe
message. `FORBIDDEN` maps to HTTP 403 and `INVALID_REQUEST` to HTTP 400.
Unexpected provider exceptions are logged and exposed as a stable sanitized
HTTP 500 response.

### Path Expansion Support

- **Home Directory**: `~` is expanded to the user's home directory
- **Environment Variables**: `$VAR` and `${VAR}` syntax are supported
- **Relative Paths**: Converted to absolute paths automatically
- **Path Validation**: Non-existent paths are skipped with a warning

### Error Handling

- Invalid paths are logged and skipped but don't affect default directories
- Non-existent directories are warned but don't block startup
- URL-like paths (s3://, nfs://, etc.) are rejected with a warning
- Default directories are always loaded regardless of external directory configuration

### Default Behavior

The default directories are always loaded:
1. Built-in skills: `src/xagent/skills/builtin/`
2. User skills: `.xagent/skills/`

If `XAGENT_EXTERNAL_SKILLS_LIBRARY_DIRS` is set, valid directories are appended to these defaults.

### Examples

#### Development Environment
```bash
# Use local skills directory for development
export XAGENT_EXTERNAL_SKILLS_LIBRARY_DIRS="~/dev/skills,./team_skills"
```

#### Production Environment
```bash
# Use shared skills directory
export XAGENT_EXTERNAL_SKILLS_LIBRARY_DIRS="/opt/xagent/skills,/shared/company_skills"
```

#### Team Collaboration
```bash
# Combine personal and team skills
export XAGENT_EXTERNAL_SKILLS_LIBRARY_DIRS="~/my_skills,~/team_skills"
```
