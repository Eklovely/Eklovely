"""The guided setup wizard engine: runs steps, handles Retry/Skip/Quit, remembers progress."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ai_neo.config import Config
from ai_neo.ollama_client import OllamaClient
from ai_neo.paths import Paths
from ai_neo.setup.checks import Status
from ai_neo.setup.probe import RealProbe, SystemProbe
from ai_neo.setup.state import SetupState
from ai_neo.ui import UI


@dataclass(frozen=True)
class StepResult:
    status: Status
    message: str


@dataclass(frozen=True)
class Step:
    id: str
    title: str
    what: str
    why: str
    troubleshooting: tuple[str, ...]
    preview: Callable[[SetupContext], list[str]]
    run: Callable[[SetupContext], StepResult]
    always_run: bool = False


ClientFactory = Callable[[Config], OllamaClient]


def default_client_factory(cfg: Config) -> OllamaClient:
    return OllamaClient(
        cfg.ollama.base_url,
        cfg.ollama.model,
        keep_alive=cfg.ollama.keep_alive,
        timeout=cfg.ollama.request_timeout,
    )


@dataclass
class SetupContext:
    ui: UI
    probe: SystemProbe
    paths: Paths
    cfg: Config
    env: Mapping[str, str]
    non_interactive: bool
    client_factory: ClientFactory = default_client_factory
    models: list[str] | None = None
    server_ok: bool = False
    facts: dict[str, Any] = field(default_factory=dict)
    results: dict[str, StepResult] = field(default_factory=dict)
    unresolved: list[str] = field(default_factory=list)

    def client(self) -> OllamaClient:
        return self.client_factory(self.cfg)

    def consent(self, question: str, *, kind: str, default: bool = True) -> bool:
        """Ask before commands, writes, downloads and API calls.

        With --non-interactive nobody can be asked: read-only probes and writing the config are
        allowed (defaults only), while downloads and paid API calls are never done.
        """
        if self.non_interactive:
            return kind in ("probe", "write")
        return self.ui.confirm(question, default=default)


class Wizard:
    def __init__(self, ctx: SetupContext, steps: Sequence[Step], state: SetupState) -> None:
        self.ctx = ctx
        self.steps = steps
        self.state = state

    def run(self) -> int:
        ui = self.ctx.ui
        total = len(self.steps)
        ui.heading("AI Neo guided setup")
        ui.say(
            f"{total} short steps. Press Enter to accept the recommended answer. Nothing is "
            "downloaded, installed or run without asking you first (except with "
            "--non-interactive, which only reads and writes the config file)."
        )
        ui.say(
            "Press Ctrl+C at any time to stop; run `ai-neo setup` again and finished steps "
            "are skipped.",
            dim=True,
        )
        if self.state.was_corrupt:
            ui.tag("WARN", "The saved setup progress could not be read, so we start from the top.")
        try:
            for number, step in enumerate(self.steps, 1):
                if self._skip_finished(number, step):
                    continue
                if not self._run_step(number, total, step):
                    return 1
        except EOFError:
            ui.say("\nInput ended. Progress is saved; run `ai-neo setup` to continue.")
            return 1
        return 1 if self.ctx.unresolved else 0

    def _skip_finished(self, number: int, step: Step) -> bool:
        record = self.state.finished(step.id)
        if record is None or step.always_run:
            return False
        ui = self.ctx.ui
        ui.heading(f"Step {number} of {len(self.steps)}: {step.title}")
        when = record.at[:16].replace("T", " ")
        if self.ctx.non_interactive:
            ui.tag("OK", f"Already done on {when}; skipping.")
            return True
        return ui.confirm(f"This step was already completed on {when}. Skip it?", default=True)

    def _run_step(self, number: int, total: int, step: Step) -> bool:
        """Run one step with retries. Returns False if the user chose to quit."""
        ui, ctx = self.ctx.ui, self.ctx
        if not (self.state.finished(step.id) and not step.always_run):
            ui.heading(f"Step {number} of {total}: {step.title}")
        ui.say(step.what)
        ui.say(step.why)
        actions = step.preview(ctx)
        if actions:
            ui.say("What will happen:")
            for action in actions:
                ui.say(f"  > {action}")
        while True:
            result = self._attempt(step)
            ctx.results[step.id] = result
            ui.tag(result.status.value, result.message)
            if result.status is not Status.FAIL:
                self._remember(step, "done", result)
                return True
            ui.say("Most likely causes and fixes:")
            for tip in step.troubleshooting:
                ui.say(f"  - {tip}")
            if ctx.non_interactive:
                self._remember(step, "failed", result)
                ctx.unresolved.append(step.id)
                return True
            choice = self._ask_choice()
            if choice == "r":
                continue
            if choice == "s":
                self._remember(step, "skipped", result)
                ctx.unresolved.append(step.id)
                ui.say("Skipped. You can finish this later with `ai-neo setup`.", dim=True)
                return True
            self._remember(step, "failed", result)
            ui.say(
                "Stopped. Your progress is saved; run `ai-neo setup` to continue where you "
                "left off."
            )
            return False

    def _attempt(self, step: Step) -> StepResult:
        try:
            return step.run(self.ctx)
        except KeyboardInterrupt:
            return StepResult(Status.FAIL, "Interrupted with Ctrl+C.")
        except EOFError:
            raise
        except Exception as exc:  # a step bug must never show a traceback to a beginner
            return StepResult(
                Status.FAIL, f"Something unexpected went wrong: {type(exc).__name__}: {exc}"
            )

    def _ask_choice(self) -> str:
        try:
            return self.ctx.ui.choose("What now? [R]etry, [S]kip, or [Q]uit:", "rsq")
        except KeyboardInterrupt:
            return "q"

    def _remember(self, step: Step, status: str, result: StepResult) -> None:
        self.state.record(step.id, status, result.status.value, result.message)
        self.state.save(self.ctx.paths.state)


def run_setup(
    paths: Paths,
    env: Mapping[str, str],
    ui: UI,
    cfg: Config,
    *,
    non_interactive: bool = False,
    reset: bool = False,
    probe: SystemProbe | None = None,
    client_factory: ClientFactory | None = None,
    steps: Sequence[Step] | None = None,
) -> int:
    from ai_neo.setup.steps import STEPS

    paths.root.mkdir(parents=True, exist_ok=True)
    if reset and paths.state.exists():
        paths.state.unlink()
    ctx = SetupContext(
        ui=ui,
        probe=probe or RealProbe(),
        paths=paths,
        cfg=cfg,
        env=env,
        non_interactive=non_interactive,
        client_factory=client_factory or default_client_factory,
    )
    state = SetupState.load(paths.state)
    return Wizard(ctx, steps or STEPS, state).run()
