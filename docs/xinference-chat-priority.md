# Xinference chat priority

When a running model advertises `supports_chat_priority: true`, Xagent
automatically includes `priority_hint: -1` in streaming and non-streaming
Xinference chat requests made by interactive commands. Xinference uses lower
values for earlier scheduling. An explicit request `priority` takes precedence.

Batch commands and automatic resumes use normal priority by default. Automatic
resumes can be classified as interactive for Xagent's own task queue while
still using normal priority in Xinference. Models without the advertised
capability receive no automatic priority hint.
