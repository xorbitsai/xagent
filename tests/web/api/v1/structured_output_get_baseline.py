"""GET /v1/chat/tasks/{id} response bodies of the tasks in BASELINE_TASKS, as
the endpoint returned them before it carried the structured output fields."""

BASELINE_BODIES = [
    '{"task_id":1,"agent_id":1,"workforce_id":null,"status":"completed","run_id":"run-1","state_version":0,"control_state":"idle","input":"question","output":"final answer","error":null,"created_at":"2026-11-01T08:30:00","completed_at":"2026-11-01T08:30:00","pending_interaction":null}',
    '{"task_id":2,"agent_id":1,"workforce_id":null,"status":"completed","run_id":"run-1","state_version":0,"control_state":"idle","input":"question","output":"with a record","error":null,"created_at":"2026-11-01T08:30:00","completed_at":"2026-11-01T08:30:00","pending_interaction":null}',
    '{"task_id":3,"agent_id":1,"workforce_id":null,"status":"failed","run_id":"run-1","state_version":0,"control_state":"idle","input":"question","output":null,"error":"agent crashed","created_at":"2026-11-01T08:30:00","completed_at":"2026-11-01T08:30:00","pending_interaction":null}',
    '{"task_id":4,"agent_id":1,"workforce_id":null,"status":"running","run_id":"run-1","state_version":0,"control_state":"idle","input":"question","output":null,"error":null,"created_at":"2026-11-01T08:30:00","completed_at":null,"pending_interaction":null}',
    '{"task_id":5,"agent_id":1,"workforce_id":null,"status":"pending","run_id":null,"state_version":0,"control_state":"idle","input":"question","output":null,"error":null,"created_at":"2026-11-01T08:30:00","completed_at":null,"pending_interaction":null}',
    '{"task_id":6,"agent_id":1,"workforce_id":null,"status":"waiting_for_user","run_id":"run-1","state_version":0,"control_state":"idle","input":"question","output":null,"error":null,"created_at":"2026-11-01T08:30:00","completed_at":null,"pending_interaction":null}',
    '{"task_id":7,"agent_id":1,"workforce_id":null,"status":"completed","run_id":"run-1","state_version":0,"control_state":"idle","input":"question","output":"\xe9 \u4e2d\u6587 \U0001f600","error":null,"created_at":"2026-11-01T08:30:00","completed_at":"2026-11-01T08:30:00","pending_interaction":null}',
]
