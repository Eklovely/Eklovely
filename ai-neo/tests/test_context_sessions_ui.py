from __future__ import annotations

import io
from pathlib import Path

import pytest

from ai_neo.context import ContextManager, estimate_message, estimate_tokens, render_transcript
from ai_neo.sessions import Session, SessionNotFound, SessionStore, export_markdown, safe_name
from ai_neo.types import GenStats, Message, StreamEvent, ToolCall
from ai_neo.ui import HAVE_RICH, PlainUI, RichUI, format_stats, human_bytes, make_ui
from tests.helpers import ScriptedUI


def chat_session(turns: int, size: int = 400) -> Session:
    session = Session(mode="reason", model="qwen3:30b")
    for i in range(turns):
        session.messages.append(Message("user", f"question {i} " + "q" * size))
        session.messages.append(Message("assistant", f"answer {i} " + "a" * size))
    return session


# -- context -------------------------------------------------------------------------------------


def test_estimate_is_chars_over_four() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2
    msg = Message("assistant", "x" * 40, tool_calls=[ToolCall("calculator", {"expression": "1+1"})])
    assert estimate_message(msg) > 14


def test_no_compaction_below_threshold() -> None:
    manager = ContextManager(num_ctx=8192, summarize_at=0.75, keep_last_turns=4)
    session = chat_session(3)
    assert not manager.needs_compaction(session)
    notes: list[str] = []
    assert not manager.compact(session, lambda p: "never", notes.append)
    assert notes == []


def test_compaction_summarizes_old_turns_and_keeps_recent() -> None:
    manager = ContextManager(num_ctx=2048, summarize_at=0.75, keep_last_turns=2)
    session = chat_session(8)
    prompts: list[str] = []

    def summarizer(prompt: str) -> str:
        prompts.append(prompt)
        return "- user asked 6 questions"

    notes: list[str] = []
    assert manager.needs_compaction(session)
    assert manager.compact(session, summarizer, notes.append)
    assert session.summary == "- user asked 6 questions"
    assert [m.role for m in session.messages] == ["user", "assistant"] * 2
    assert session.messages[0].content.startswith("question 6")
    assert "question 0" in prompts[0]
    assert "summarized 12 older messages" in notes[0]
    assert "- user asked 6 questions" in session.system_prompt()


def test_second_compaction_includes_previous_summary() -> None:
    manager = ContextManager(num_ctx=2048, summarize_at=0.75, keep_last_turns=2)
    session = chat_session(8)
    session.summary = "EARLIER SUMMARY"
    prompts: list[str] = []
    manager.compact(session, lambda p: prompts.append(p) or "new", lambda n: None)
    assert "EARLIER SUMMARY" in prompts[0]


def test_failed_summarizer_still_frees_space() -> None:
    manager = ContextManager(num_ctx=2048, summarize_at=0.75, keep_last_turns=2)
    session = chat_session(8)
    manager.compact(session, lambda p: "", lambda n: None)
    assert "dropped" in session.summary and len(session.messages) == 4


def test_big_tool_outputs_are_shortened_when_nothing_to_summarize() -> None:
    manager = ContextManager(num_ctx=1024, summarize_at=0.5, keep_last_turns=4)
    session = Session(mode="code", model="m")
    session.messages = [
        Message("user", "run it"),
        Message("assistant", "", tool_calls=[ToolCall("run_python", {"code": "x"})]),
        Message("tool", "o" * 6000, tool_name="run_python"),
        Message("assistant", "done"),
    ]
    notes: list[str] = []
    assert manager.compact(session, lambda p: "", notes.append)
    assert len(session.messages[2].content) < 900 and "shortened" in notes[0]


def test_transcript_is_clipped() -> None:
    text = render_transcript([Message("user", "u" * 5000)], max_chars=2500)
    assert len(text) < 2600


# -- sessions ------------------------------------------------------------------------------------


def test_session_roundtrip_with_tool_calls(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    session = chat_session(1, 10)
    session.messages.append(
        Message("assistant", "", tool_calls=[ToolCall("calculator", {"expression": "2+2"})])
    )
    session.messages.append(Message("tool", "2+2 = 4", tool_name="calculator"))
    session.summary = "earlier stuff"
    session.system_override = "Be brief."
    path = store.save(session, "My Chat")
    assert path.parent == tmp_path / "sessions" and path.name == "My_Chat.json"
    loaded = store.load("My Chat")
    assert loaded.messages == session.messages
    assert loaded.summary == "earlier stuff" and loaded.system_override == "Be brief."
    assert store.list()[0][0] == "My_Chat"


def test_load_latest_and_missing(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "s")
    with pytest.raises(SessionNotFound):
        store.load()
    first, second = chat_session(1, 5), chat_session(2, 5)
    store.save(first, "first")
    store.save(second, "second")
    assert store.load().id == second.id
    with pytest.raises(SessionNotFound):
        store.load("ghost")
    (tmp_path / "s" / "broken.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(SessionNotFound, match="not a readable"):
        store.load("broken")


def test_session_names_cannot_escape_the_directory(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    path = store.save(chat_session(1, 5), "../../evil")
    assert path.parent == tmp_path / "sessions"
    assert safe_name("a b/c\\d") == "a_b_c_d"
    assert safe_name("...") == "session"


def test_names_keep_other_languages_and_avoid_windows_device_names() -> None:
    japanese, chinese = "\u65e5\u672c\u8a9e", "\u4e2d\u6587"
    assert safe_name(japanese) == japanese and safe_name(chinese) == chinese
    assert safe_name(japanese) != safe_name(chinese)  # they used to collapse into "session"
    assert safe_name("caf\u00e9 notes") == "caf\u00e9_notes"
    for device in ("con", "NUL", "com1", "Lpt9", "aux.txt", "PRN"):
        assert safe_name(device).startswith("_")
    assert safe_name("console") == "console"  # only the exact reserved names
    assert safe_name("a\x00b\x1fc\u202ed") == "a_b_c_d"  # control and direction characters go
    assert safe_name("x" * 200) == "x" * 60


def test_markdown_export() -> None:
    session = chat_session(1, 5)
    session.name = "demo"
    session.messages.append(
        Message("assistant", "", tool_calls=[ToolCall("calculator", {"expression": "1+1"})])
    )
    session.messages.append(Message("tool", "1+1 = 2", tool_name="calculator"))
    md = export_markdown(session)
    assert md.startswith("# AI Neo session demo")
    assert "## You" in md and "## AI Neo" in md and "Tool result: calculator" in md
    assert '"expression": "1+1"' in md


# -- ui ------------------------------------------------------------------------------------------


def test_confirm_and_choose_handle_bad_input() -> None:
    ui = ScriptedUI(["maybe", "", "Y", "x", "S"])
    assert ui.confirm("Continue?", default=False) is False  # "maybe" rejected, "" -> default
    assert ui.confirm("Really?", default=False) is True
    assert ui.choose("Retry, skip, quit?", "rsq") == "s"
    assert "Please answer y or n." in ui.text and "Please type one of" in ui.text


def test_ask_default() -> None:
    ui = ScriptedUI(["", "typed"])
    assert ui.ask("Folder", "~/work") == "~/work"
    assert ui.ask("Folder", "~/work") == "typed"
    assert "[~/work]" in ui.prompts[0]


def test_plain_stream_separates_thinking_from_answer() -> None:
    ui = ScriptedUI()
    stream = ui.stream()
    stream.event(StreamEvent("thinking", "hmm "))
    stream.event(StreamEvent("thinking", "ok"))
    stream.event(StreamEvent("answer", "Hello"))
    stream.close()
    assert ui.text == "(thinking) hmm ok\n--- answer ---\nHello\n"


def test_plain_ui_color_when_asked() -> None:
    ui = PlainUI(out=io.StringIO(), color=True)
    ui.tag("OK", "fine")
    assert "\x1b[32m[OK]" in ui.out.getvalue()  # type: ignore[attr-defined]


def test_formatting_helpers() -> None:
    assert human_bytes(512) == "512 B"
    assert human_bytes(19 * 1024**3) == "19.0 GB"
    stats = GenStats(
        eval_count=100,
        eval_duration_ns=10_000_000_000,
        total_duration_ns=12_000_000_000,
        load_duration_ns=5_000_000_000,
    )
    line = format_stats(stats)
    assert "10.0 tok/s" in line and "100 tokens" in line and "model load 5.0s" in line


def test_plain_table_and_progress_nontty() -> None:
    ui = ScriptedUI()
    ui.table(["Item", "Status"], [["python", "PASS"], ["ollama", "FAIL"]], title="Checks")
    bar = ui.progress("pulling")
    for done in (10, 50, 100):
        bar.update(done, 100, "pulling x")
    bar.close()
    assert "python" in ui.text and "100%" in ui.text


@pytest.mark.skipif(not HAVE_RICH, reason="rich not installed")
def test_rich_ui_smoke() -> None:
    from rich.console import Console

    buffer = io.StringIO()
    ui = RichUI(Console(file=buffer, force_terminal=False, width=100))
    ui.heading("Step 1 of 9: System check")
    ui.tag("WARN", "low ram")
    ui.preview("Command", "ollama pull qwen3:30b")
    ui.table(["a", "b"], [["1", "2"]], title="t")
    stream = ui.stream()
    stream.event(StreamEvent("thinking", "pondering\nmore"))
    stream.event(StreamEvent("answer", "The answer"))
    stream.close()
    bar = ui.progress("download")
    bar.update(5, 10, "layer")
    bar.close()
    out = buffer.getvalue()
    assert "System check" in out and "[WARN]" in out and "The answer" in out
    assert "thought for" in out


def test_make_ui_falls_back_to_plain_when_not_a_tty() -> None:
    assert isinstance(make_ui(), PlainUI)
    assert isinstance(make_ui(plain=True), PlainUI)
