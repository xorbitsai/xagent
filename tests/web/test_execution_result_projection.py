from xagent.core.agent.execution_adapter import INTERRUPTED_USER_MESSAGE
from xagent.web.models.task import TaskStatus
from xagent.web.services.execution_result_projection import (
    EMPTY_CHANNEL_OUTPUT_FALLBACK,
    project_execution_result_for_channel,
)


def test_project_execution_result_waiting_for_user_uses_chat_message_as_question():
    projection = project_execution_result_for_channel(
        {
            "status": "waiting_for_user",
            "success": False,
            "output": "Need input.",
            "chat_response": {"message": "Choose A or B", "interactions": []},
        }
    )

    assert projection.task_status == TaskStatus.WAITING_FOR_USER
    assert projection.visible_text == "Choose A or B"
    assert projection.transcript_content == "Choose A or B"
    assert projection.message_type == "question"
    assert projection.interactions == []


def test_project_execution_result_appends_interactions_to_visible_text():
    projection = project_execution_result_for_channel(
        {
            "success": True,
            "output": "Need details.",
            "chat_response": {
                "message": "Choose a destination",
                "interactions": [
                    {
                        "label": "Destination",
                        "options": [{"label": "Tokyo"}, {"value": "Osaka"}],
                    }
                ],
            },
        }
    )

    assert projection.task_status == TaskStatus.COMPLETED
    assert projection.transcript_content == "Choose a destination"
    assert projection.message_type == "question"
    assert projection.interactions == [
        {
            "label": "Destination",
            "options": [{"label": "Tokyo"}, {"value": "Osaka"}],
        }
    ]
    assert projection.visible_text == (
        "Choose a destination\n\n• Destination\n  Options: Tokyo, Osaka"
    )


def test_project_execution_result_falls_back_for_empty_output():
    projection = project_execution_result_for_channel({"success": True, "output": None})

    assert projection.task_status == TaskStatus.COMPLETED
    assert projection.visible_text == EMPTY_CHANNEL_OUTPUT_FALLBACK
    assert projection.transcript_content == EMPTY_CHANNEL_OUTPUT_FALLBACK
    assert projection.message_type == "assistant_message"


def test_project_execution_result_maps_interrupted_to_paused():
    projection = project_execution_result_for_channel(
        {
            "status": "interrupted",
            "success": False,
            "output": "ReActPattern interrupted.",
        }
    )

    assert projection.task_status == TaskStatus.PAUSED
    assert projection.visible_text == INTERRUPTED_USER_MESSAGE
    assert projection.transcript_content == ""
    assert projection.interactions == []
