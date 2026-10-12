from __future__ import annotations

from pathlib import Path

from ai_neo.agent import extract_text_tool_calls, resolve_think, sampling_for
from ai_neo.config import default_config
from ai_neo.prompts import MODES
from ai_neo.sessions import Session
from ai_neo.types import StreamEvent
from tests.fake_ollama import ChatReply, FakeOllama, text, tool_call
from tests.helpers import ScriptedUI, make_agent


def new_session(mode: str = "reason") -> Session:
    return Session(mode=mode, model="qwen3:30b")


def test_plain_answer(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    fake_ollama.queue(text("Paris.", eval_count=10, eval_duration=1_000_000_000))
    agent, ui, _ = make_agent(fake_ollama, tmp_path)
    session = new_session()
    result = agent.run_turn(session, "Capital of France?", think=False)
    assert result.answer == "Paris."
    assert result.stats.tok_per_s == 10 and result.steps == 1
    assert [m.role for m in session.messages] == ["user", "assistant"]
    assert session.turns == 1 and session.generated_tokens == 10
    assert "Paris." in ui.text
    payload = fake_ollama.chat_payloads()[0]
    assert payload["messages"][0]["role"] == "system"
    assert payload["options"]["temperature"] == 0.7  # non-thinking preset


def test_thinking_is_shown_but_never_stored(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    fake_ollama.queue(ChatReply(thinking=["Considering ", "options."], content=["Answer."]))
    agent, ui, _ = make_agent(fake_ollama, tmp_path)
    session = new_session()
    result = agent.run_turn(session, "q", think=True)
    assert result.thinking == "Considering options." and result.answer == "Answer."
    assert "(thinking) Considering options." in ui.text and "--- answer ---" in ui.text
    assert all("Considering" not in m.content for m in session.messages)
    assert fake_ollama.chat_payloads()[0]["options"]["temperature"] == 0.6  # thinking preset
    assert fake_ollama.chat_payloads()[0]["think"] is True


def test_tool_loop_runs_calculator_and_feeds_result_back(
    fake_ollama: FakeOllama, tmp_path: Path
) -> None:
    fake_ollama.queue(tool_call("calculator", {"expression": "2+2*3"}), text("It is 8."))
    agent, _, _ = make_agent(fake_ollama, tmp_path)
    session = new_session()
    result = agent.run_turn(session, "what is 2+2*3?", think=False)
    assert result.answer == "It is 8." and result.steps == 2
    assert [(e.name, e.ok) for e in result.tool_events] == [("calculator", True)]
    second = fake_ollama.chat_payloads()[1]["messages"]
    tool_message = second[-1]
    assert tool_message["role"] == "tool" and tool_message["tool_name"] == "calculator"
    assert tool_message["content"] == "2+2*3 = 8"
    assistant_call = second[-2]["tool_calls"][0]["function"]
    assert assistant_call == {"name": "calculator", "arguments": {"expression": "2+2*3"}}
    assert "tools" in fake_ollama.chat_payloads()[0]


def test_malformed_tool_arguments_are_returned_to_the_model(
    fake_ollama: FakeOllama, tmp_path: Path
) -> None:
    fake_ollama.queue(
        tool_call("calculator", "{not json"),
        tool_call("nonexistent_tool", {"x": 1}),
        tool_call("calculator", {"expression": "1+1"}),
        text("2"),
    )
    agent, _, _ = make_agent(fake_ollama, tmp_path)
    result = agent.run_turn(new_session(), "1+1?", think=False)
    assert result.answer == "2"
    assert [e.ok for e in result.tool_events] == [False, False, True]
    assert "not valid JSON" in result.tool_events[0].output
    assert "unknown tool" in result.tool_events[1].output


def test_server_side_tool_parse_error_triggers_retry(
    fake_ollama: FakeOllama, tmp_path: Path
) -> None:
    fake_ollama.queue(
        ChatReply(status=500, error="error parsing tool call: raw='{oops' err=bad json"),
        tool_call("calculator", {"expression": "5*5"}),
        text("25"),
    )
    agent, ui, _ = make_agent(fake_ollama, tmp_path)
    session = new_session()
    result = agent.run_turn(session, "5*5?", think=False)
    assert result.answer == "25" and result.error is None
    retry_note = fake_ollama.chat_payloads()[1]["messages"][-1]
    assert retry_note["role"] == "user" and "could not be parsed" in retry_note["content"]
    assert "malformed tool call" in ui.text


def test_step_limit_is_hard_and_last_call_has_no_tools(
    fake_ollama: FakeOllama, tmp_path: Path
) -> None:
    fake_ollama.chat_handler = lambda payload: (
        text("Best answer without more tools.")
        if "tools" not in payload
        else tool_call("calculator", {"expression": "1+1"})
    )
    agent, _, _ = make_agent(fake_ollama, tmp_path, **{"tools.max_steps": 3})
    result = agent.run_turn(new_session(), "loop forever", think=False)
    payloads = fake_ollama.chat_payloads()
    assert len(payloads) == 3 and result.steps == 3
    assert "tools" in payloads[0] and "tools" in payloads[1] and "tools" not in payloads[2]
    assert "Do not call any more tools" in payloads[2]["messages"][-1]["content"]
    assert result.answer == "Best answer without more tools."


def test_step_limit_flag_when_model_insists_on_tools(
    fake_ollama: FakeOllama, tmp_path: Path
) -> None:
    fake_ollama.chat_handler = lambda payload: tool_call("calculator", {"expression": "1+1"})
    agent, _, _ = make_agent(fake_ollama, tmp_path, **{"tools.max_steps": 2})
    result = agent.run_turn(new_session(), "x", think=False)
    assert len(fake_ollama.chat_payloads()) == 2
    assert result.step_limit_hit and result.answer == ""


def test_text_embedded_tool_calls_are_understood(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    body = '<tool_call>{"name": "calculator", "arguments": {"expression": "3*3"}}</tool_call>'
    fake_ollama.queue(text(body), text("9"))
    agent, _, _ = make_agent(fake_ollama, tmp_path)
    result = agent.run_turn(new_session(), "3*3?", think=False)
    assert result.answer == "9" and result.tool_events[0].output == "3*3 = 9"


def test_extract_text_tool_calls_malformed() -> None:
    calls, cleaned = extract_text_tool_calls("hi <tool_call>{broken</tool_call> there")
    assert calls[0].name == "" and "tool_call" not in cleaned
    assert extract_text_tool_calls("no calls here") == ([], "no calls here")


def test_declined_tool_is_reported_to_model(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    fake_ollama.queue(
        tool_call("write_file", {"path": "a.txt", "content": "x"}), text("ok, not written")
    )
    ui = ScriptedUI(["n"])
    agent, _, cfg = make_agent(fake_ollama, tmp_path, ui=ui, yes=False)
    session = new_session("code")
    result = agent.run_turn(session, "write a file", think=False)
    assert result.tool_events[0].meta["declined"] is True
    assert not (cfg.workspace / "a.txt").exists()
    assert "declined" in fake_ollama.chat_payloads()[1]["messages"][-1]["content"]


def test_ctrl_c_during_generation_keeps_partial_and_does_not_raise(
    fake_ollama: FakeOllama, tmp_path: Path
) -> None:
    fake_ollama.queue(text("one ", "two ", "three ", "four"))
    ui = ScriptedUI()
    original = ui.stream

    def stream():  # type: ignore[no-untyped-def]
        renderer = original()
        seen = []

        class Interrupting:
            def event(self, event: StreamEvent) -> None:
                renderer.event(event)
                seen.append(event)
                if len(seen) == 2:
                    raise KeyboardInterrupt

            def close(self) -> None:
                renderer.close()

        return Interrupting()

    ui.stream = stream  # type: ignore[method-assign]
    agent, _, _ = make_agent(fake_ollama, tmp_path, ui=ui)
    session = new_session()
    result = agent.run_turn(session, "count", think=False)
    assert result.interrupted and result.answer == "one two"
    assert session.messages[-1].role == "assistant" and session.messages[-1].content == "one two"
    # the session is still usable afterwards
    fake_ollama.queue(text("fine"))
    assert agent.run_turn(session, "again", think=False).answer == "fine"


def test_ollama_errors_become_result_errors(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    fake_ollama.queue(text("partial", stream_error="runner crashed"))
    agent, _, _ = make_agent(fake_ollama, tmp_path)
    session = new_session()
    result = agent.run_turn(session, "q", think=False)
    assert result.error and "runner crashed" in result.error
    assert result.answer == "partial"


def test_history_is_summarized_near_the_limit(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    agent, ui, _ = make_agent(
        fake_ollama, tmp_path, **{"ollama.num_ctx": 1024, "context.keep_last_turns": 1}
    )
    session = new_session()
    for i in range(3):
        session.messages.append(
            __import__("ai_neo.types", fromlist=["Message"]).Message(
                "user", f"old {i} " + "x" * 600
            )
        )
        session.messages.append(
            __import__("ai_neo.types", fromlist=["Message"]).Message("assistant", "y" * 600)
        )
    fake_ollama.queue(text("- summary of the old stuff"), text("final answer"))
    result = agent.run_turn(session, "new question", think=False)
    assert result.answer == "final answer"
    assert session.summary == "- summary of the old stuff"
    assert "summarized" in ui.text
    summarize_payload = fake_ollama.chat_payloads()[0]
    assert summarize_payload["think"] is False and "tools" not in summarize_payload
    system = fake_ollama.chat_payloads()[1]["messages"][0]["content"]
    assert "summary of the old stuff" in system


def test_resolve_think_and_sampling() -> None:
    preset = MODES["write"]
    assert resolve_think("auto", None, preset) is False
    assert resolve_think("auto", None, MODES["reason"]) is True
    assert resolve_think("off", None, MODES["reason"]) is False
    assert resolve_think("off", True, MODES["reason"]) is True
    cfg = default_config()
    thinking = sampling_for(cfg, preset, True)
    plain = sampling_for(cfg, preset, False)
    assert (thinking.temperature, thinking.top_p, thinking.top_k) == (0.6, 0.95, 20)
    assert (plain.temperature, plain.top_p, plain.top_k) == (0.7, 0.8, 20)
    assert thinking.num_ctx == 8192
    assert all(m.sampling(t).temperature > 0 for m in MODES.values() for t in (True, False))
