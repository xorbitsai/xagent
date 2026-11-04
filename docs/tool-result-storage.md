# Stored tool results

When a tool returns a result that is too long for the model's context, xagent
can store the long part in a file in the task workspace instead of cutting it
off. The model sees a short placeholder and a notice naming the stored file,
and reads the file back with the `read_tool_result` tool.

## When a result is stored

Storing is enabled for a tool set only when it contains a `read_tool_result`
tool bound to a real task workspace. A tool set without one -- no file tools,
the tool-listing endpoint's mock workspace, or a tool policy that removes
`read_tool_result` -- keeps the ordinary behavior: every string longer than
the limit is truncated. Building such a tool set logs one INFO line starting
with `Tool result spill disabled:`.

The limit is `XAGENT_TOOL_MAX_OUTPUT_LENGTH` (characters, default 51200).
Only dict results are stored. For a dict result:

- Each value inside it whose JSON form is longer than the limit is stored in
  its own file and replaced by a placeholder, at most
  `SPILL_MAX_FILES_PER_RESULT` (8) files per result.
- If no single value is over the limit but the whole result is, the whole
  result is stored in one file. Every key is kept; a value longer than the
  placeholder is replaced by it, except fields listed in
  `SPILL_ENVELOPE_KEYS` (such as `status`, `error` and `output`).
- Results that wait for user input, classified tool failures and file
  references are never stored. Binary values are left to ordinary filtering.

One tool set stores at most `SPILL_MAX_FILES_PER_RUN` (64) files; after
that, over-long values are truncated as before. One file holds at most
`SPILL_MAX_FILE_BYTES` (8 MiB); a longer value is cut at an item or line
boundary, and the notice says how many items were stored.

## Where files go

Files are written to `output/tool-results/` inside the task workspace, named
`<tool name>-<digest>.json` for arrays and objects and
`<tool name>-<digest>.txt` for everything else. The tool-name prefix is
reduced to letters, digits, `_` and `-` and cut to 64 characters; the digest
is the first 32 hex characters of the SHA-256 of the file's bytes. The model
refers to a file as `tool-results/<name>`. The directory belongs to the task,
so a later run of the same task can list files an earlier run stored. These
files are not part of the task's output file list.

## Reading a stored result back

The ReAct pattern offers `read_tool_result` to the model once the run has
registered at least one stored result. It has three uses:

- No `path`: list the stored files by path and size, a page of at most
  `SPILL_MAX_FILES_PER_RUN` entries; `start` and `end` pick entry numbers.
- `path` only: read the whole stored result.
- `path` with `start` and `end`: read items `start` to `end` (1-based,
  inclusive). An item is an array element, a top-level object entry, or a
  line of text, depending on the file's content. `offset` is a 0-based
  character position inside the selected text, used to continue reading an
  item longer than one reply.

One reply returns at most the smaller of `SPILL_READ_MAX_CHARS` (12000) and
`XAGENT_TOOL_MAX_OUTPUT_LENGTH` characters, and the tool description states
that number. A longer selection returns a preview that says where it starts,
how long the whole text is, and whether more follows. A file that is gone, or
whose bytes no longer match the digest in its name, is reported as
unavailable.

The reply goes through the same output filter as every other tool and is
never stored again. For any positive limit the text of one reply, in `output`
or `content_preview`, is never longer than the output limit, so the filter
does not cut it, and advancing `offset` by the number in the tool description
continues exactly where the previous reply ended. The other fields of a reply
can still be cut; see Known limitations.

## Settings and limits

- `XAGENT_TOOL_MAX_OUTPUT_LENGTH` (environment, see `example.env`): the
  per-string truncation limit and the storing threshold. Below
  `SPILL_READ_MAX_CHARS` it is also the size of one `read_tool_result` reply.
- `SPILL_MAX_FILES_PER_RESULT`, `SPILL_MAX_FILES_PER_RUN`,
  `SPILL_MAX_FILE_BYTES` and `SPILL_READ_MAX_CHARS` are constants in
  `src/xagent/core/tools/tool_result_spill.py`, not settings.

## Known limitations

- Only dict results are stored. A tool that returns a plain string or a list
  is truncated as before.
- The output filter still cuts the short string fields of a `read_tool_result`
  preview when the output limit is shorter than the field: `instruction` (160
  characters) under a limit below 160, and `relative_path` (at most 130
  characters) under a limit below its own length.
- A listing page holds up to `SPILL_MAX_FILES_PER_RUN` (64) entries, and the
  output filter also caps every list at `XAGENT_TOOL_MAX_FIELD_COUNT` items. A
  deployment that sets that count below 64 gets a truncated listing.
