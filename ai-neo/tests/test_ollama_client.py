from __future__ import annotations

import threading
import time

import pytest

from ai_neo.ollama_client import (
    ModelNotFound,
    OllamaClient,
    OllamaError,
    OllamaUnavailable,
    PullCancelled,
    PullFailed,
    PullProgress,
    ThinkTagSplitter,
    ToolCallParseError,
    model_installed,
)
from ai_neo.types import Message, Sampling, StreamEvent
from tests.fake_ollama import ChatReply, FakeOllama, PullScript, text, tool_call

SAMPLING = Sampling(temperature=0.6, top_p=0.95, top_k=20, num_ctx=8192, num_predict=512)
HELLO = [Message("user", "hi")]


def make_client(server: FakeOllama, **kw: object) -> OllamaClient:
    return OllamaClient(server.url, "qwen3:30b", keep_alive="30m", timeout=10, **kw)  # type: ignore[arg-type]


def test_streams_answer_and_stats(fake_ollama: FakeOllama) -> None:
    fake_ollama.queue(text("Hel", "lo ", "there", eval_count=30, eval_duration=3_000_000_000))
    events: list[StreamEvent] = []
    with make_client(fake_ollama) as client:
        result = client.chat(HELLO, sampling=SAMPLING, on_event=events.append)
    assert result.content == "Hello there"
    assert [e.text for e in events] == ["Hel", "lo ", "there"]
    assert result.stats.eval_count == 30
    assert result.stats.tok_per_s == pytest.approx(10.0)
    assert result.stats.total_duration_ns == 2_500_000_000
    assert not result.interrupted


def test_request_payload_has_options_keep_alive_and_stream(fake_ollama: FakeOllama) -> None:
    fake_ollama.queue(text("ok"))
    with make_client(fake_ollama) as client:
        client.chat(HELLO, sampling=SAMPLING, think=True, tools=[{"type": "function"}])
    payload = fake_ollama.chat_payloads()[0]
    assert payload["stream"] is True
    assert payload["keep_alive"] == "30m"
    assert payload["think"] is True
    assert payload["options"] == {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "num_ctx": 8192,
        "num_predict": 512,
    }
    assert payload["tools"] == [{"type": "function"}]


def test_thinking_field_is_separated(fake_ollama: FakeOllama) -> None:
    fake_ollama.queue(ChatReply(thinking=["Let me ", "think."], content=["\n\nAnswer."]))
    events: list[StreamEvent] = []
    with make_client(fake_ollama) as client:
        result = client.chat(HELLO, sampling=SAMPLING, think=True, on_event=events.append)
    assert result.thinking == "Let me think."
    assert result.content == "Answer."
    assert [e.kind for e in events] == ["thinking", "thinking", "answer"]


def test_think_tags_in_content_are_separated_across_chunks(fake_ollama: FakeOllama) -> None:
    fake_ollama.queue(text("<thi", "nk>pondering", " deeply</th", "ink>\n\nThe ", "answer"))
    with make_client(fake_ollama) as client:
        result = client.chat(HELLO, sampling=SAMPLING)
    assert result.thinking == "pondering deeply"
    assert result.content == "The answer"


def test_thinking_never_goes_back_into_history() -> None:
    message = Message("assistant", "answer")
    assert "thinking" not in message.to_ollama()
    assert not hasattr(message, "thinking")


def test_splitter_handles_unclosed_think_and_lookalikes() -> None:
    splitter = ThinkTagSplitter()
    events = splitter.feed("a < b and <thin")
    events += splitter.flush()
    assert "".join(e.text for e in events) == "a < b and <thin"
    assert {e.kind for e in events} == {"answer"}


def test_400_on_think_retries_once_without_it(fake_ollama: FakeOllama) -> None:
    fake_ollama.queue(
        ChatReply(status=400, error='"qwen3:30b" does not support thinking'),
        text("plain answer"),
    )
    with make_client(fake_ollama) as client:
        result = client.chat(HELLO, sampling=SAMPLING, think=True)
        assert result.content == "plain answer"
        assert "think" in fake_ollama.chat_payloads()[0]
        assert "think" not in fake_ollama.chat_payloads()[1]
        fake_ollama.queue(text("second"))
        client.chat(HELLO, sampling=SAMPLING, think=True)  # remembered: no further 400
    assert "think" not in fake_ollama.chat_payloads()[2]


def test_other_400_is_not_retried(fake_ollama: FakeOllama) -> None:
    fake_ollama.queue(ChatReply(status=400, error="invalid options"))
    with make_client(fake_ollama) as client, pytest.raises(OllamaError, match="invalid options"):
        client.chat(HELLO, sampling=SAMPLING, think=True)
    assert len(fake_ollama.chat_payloads()) == 1


def test_404_model_missing_has_pull_hint(fake_ollama: FakeOllama) -> None:
    with OllamaClient(fake_ollama.url, "nope:7b") as client:
        with pytest.raises(ModelNotFound) as info:
            client.chat(HELLO, sampling=SAMPLING)
        assert "ollama pull nope:7b" in str(info.value)
        with pytest.raises(ModelNotFound):
            client.health()


def test_health_ok_and_model_matching(fake_ollama: FakeOllama) -> None:
    with make_client(fake_ollama) as client:
        assert client.health() == ["qwen3:30b"]
    assert model_installed(["qwen3:latest"], "qwen3")
    assert not model_installed(["qwen3:8b"], "qwen3:30b")


def test_server_down_gives_actionable_error() -> None:
    client = OllamaClient("http://127.0.0.1:9", "qwen3:30b", timeout=2)
    with pytest.raises(OllamaUnavailable) as info:
        client.health()
    assert "ollama serve" in str(info.value)
    with pytest.raises(OllamaUnavailable):
        client.chat(HELLO, sampling=SAMPLING)


def test_ctrl_c_keeps_partial_text(fake_ollama: FakeOllama) -> None:
    fake_ollama.queue(text("one ", "two ", "three ", "four"))
    seen: list[str] = []

    def sink(event: StreamEvent) -> None:
        seen.append(event.text)
        if len(seen) == 2:
            raise KeyboardInterrupt

    with make_client(fake_ollama) as client:
        result = client.chat(HELLO, sampling=SAMPLING, on_event=sink)
    assert result.interrupted
    assert result.content == "one two"
    assert result.stats.eval_count >= 1


def test_native_tool_call_is_parsed(fake_ollama: FakeOllama) -> None:
    fake_ollama.queue(tool_call("calculator", {"expression": "2+2"}))
    with make_client(fake_ollama) as client:
        result = client.chat(HELLO, sampling=SAMPLING, tools=[{"type": "function"}])
    assert [(c.name, c.arguments) for c in result.tool_calls] == [
        ("calculator", {"expression": "2+2"})
    ]


def test_tool_call_parse_error_from_server(fake_ollama: FakeOllama) -> None:
    fake_ollama.queue(
        ChatReply(status=500, error="error parsing tool call: raw='{bad' err=invalid character")
    )
    with make_client(fake_ollama) as client, pytest.raises(ToolCallParseError):
        client.chat(HELLO, sampling=SAMPLING)


def test_mid_stream_error_keeps_partial(fake_ollama: FakeOllama) -> None:
    fake_ollama.queue(text("partial ", "text", stream_error="llama runner crashed"))
    with make_client(fake_ollama) as client, pytest.raises(OllamaError, match="crashed") as info:
        client.chat(HELLO, sampling=SAMPLING)
    assert info.value.partial is not None
    assert info.value.partial.content == "partial text"


def test_pull_streams_progress(fake_ollama: FakeOllama) -> None:
    fake_ollama.models.clear()
    fake_ollama.pull_scripts.append(
        PullScript(
            [
                {"status": "pulling manifest"},
                {"status": "pulling abc", "digest": "abc", "total": 100, "completed": 40},
                {"status": "pulling abc", "digest": "abc", "total": 100, "completed": 100},
                {"status": "success"},
            ]
        )
    )
    seen: list[PullProgress] = []
    with make_client(fake_ollama) as client:
        client.pull("qwen3:30b", seen.append)
        assert client.health() == ["qwen3:30b"]
    assert seen[1].completed == 40 and seen[1].total == 100
    assert seen[-1].status == "success"


def test_pull_interrupted_by_dropped_connection(fake_ollama: FakeOllama) -> None:
    fake_ollama.pull_scripts.append(
        PullScript([{"status": "pulling abc", "total": 100, "completed": 10}], add_model=False)
    )
    with make_client(fake_ollama) as client, pytest.raises(PullFailed, match="closed"):
        client.pull("big:model", lambda p: None)


def test_pull_error_line_and_ctrl_c(fake_ollama: FakeOllama) -> None:
    fake_ollama.pull_scripts.append(PullScript([{"error": "no space left on device"}], False))
    with make_client(fake_ollama) as client:
        with pytest.raises(PullFailed, match="no space"):
            client.pull("x:1", lambda p: None)
        fake_ollama.pull_scripts.append(PullScript([{"status": "pulling a", "total": 5}], False))

        def cancel(progress: PullProgress) -> None:
            raise KeyboardInterrupt

        with pytest.raises(PullCancelled, match="resumes"):
            client.pull("x:1", cancel)


# -- cancel(): the browser's Stop button -----------------------------------------------------------


def run_chat_in_thread(client: OllamaClient) -> tuple[list, threading.Thread]:
    box: list = []
    thread = threading.Thread(
        target=lambda: box.append(client.chat(HELLO, sampling=SAMPLING)), daemon=True
    )
    thread.start()
    return box, thread


def test_cancel_wakes_a_request_still_waiting_for_the_first_token(fake_ollama: FakeOllama) -> None:
    release = threading.Event()

    def model_is_loading(payload: dict) -> ChatReply:
        release.wait(30)  # Ollama sends nothing at all while it loads a model
        return text("too late")

    fake_ollama.chat_handler = model_is_loading
    with make_client(fake_ollama) as client:
        box, thread = run_chat_in_thread(client)
        time.sleep(0.4)
        started = time.monotonic()
        client.cancel()
        thread.join(timeout=5)
        release.set()
        assert not thread.is_alive() and time.monotonic() - started < 3
        assert box[0].interrupted and box[0].content == ""


def test_cancel_in_the_middle_of_a_reply_keeps_what_arrived(fake_ollama: FakeOllama) -> None:
    fake_ollama.queue(ChatReply(content=[f"w{i} " for i in range(100)], delay=0.03))
    with make_client(fake_ollama) as client:
        box, thread = run_chat_in_thread(client)
        time.sleep(0.4)
        client.cancel()
        thread.join(timeout=5)
        assert not thread.is_alive()
        result = box[0]
        assert result.interrupted
        assert result.content.startswith("w0 ") and len(result.content) < 100 * 4


def test_a_cancel_does_not_poison_the_next_request(fake_ollama: FakeOllama) -> None:
    fake_ollama.queue(ChatReply(content=["a "] * 50, delay=0.03), text("second answer"))
    with make_client(fake_ollama) as client:
        box, thread = run_chat_in_thread(client)
        time.sleep(0.3)
        client.cancel()
        thread.join(timeout=5)
        assert box[0].interrupted
        again = client.chat(HELLO, sampling=SAMPLING)
        assert again.content == "second answer" and not again.interrupted


def test_cancel_with_nothing_running_is_harmless(fake_ollama: FakeOllama) -> None:
    fake_ollama.queue(text("fine"))
    with make_client(fake_ollama) as client:
        client.cancel()
        assert client.chat(HELLO, sampling=SAMPLING).content == "fine"


def test_a_real_connection_failure_is_still_an_error_not_an_interruption(
    fake_ollama: FakeOllama,
) -> None:
    fake_ollama.stop()
    with make_client(fake_ollama) as client, pytest.raises(OllamaUnavailable):
        client.chat(HELLO, sampling=SAMPLING)
    fake_ollama.start = lambda: None  # the fixture's stop() runs again at teardown
