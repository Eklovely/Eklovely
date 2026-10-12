"""Command-line interface: the chat REPL, one-shot `ask`, and the other sub-commands."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ai_neo import __version__
from ai_neo.agent import Agent, resolve_think
from ai_neo.config import (
    Config,
    ConfigError,
    default_config,
    load_config,
    render_toml,
)
from ai_neo.context import ContextManager
from ai_neo.evals import (
    DEFAULT_TARGET,
    PipelineRunner,
    SuiteError,
    load_suite,
    render_report,
    run_suite,
    write_report,
)
from ai_neo.governor import Governor
from ai_neo.logging_store import LogStore
from ai_neo.ollama_client import (
    ModelNotFound,
    OllamaClient,
    OllamaError,
    OllamaUnavailable,
    missing_model_message,
)
from ai_neo.paths import Paths
from ai_neo.pipeline import TaskOptions, TaskOutcome, TaskPipeline
from ai_neo.prompts import MODES
from ai_neo.sessions import Session, SessionNotFound, SessionStore, export_markdown, safe_name
from ai_neo.tools import ToolRegistry, Workspace
from ai_neo.ui import UI, make_ui

EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_INTERRUPTED = 0, 1, 2, 130

HELP_ROWS = [
    ("/help", "show this list"),
    ("/new", "start a fresh conversation"),
    ("/think on|off|auto", "turn step-by-step thinking on or off (auto = the mode's default)"),
    ("/mode reason|compute|code|write|build", "switch what AI Neo is doing"),
    ("/model <name>", "use another installed Ollama model"),
    ("/system <text>", "set a custom system prompt (/system reset to undo, no text to show)"),
    ("/tools on|off", "allow or forbid the model to call tools"),
    ("/govern on|off", "optional: let Claude critique answers (sends text to Anthropic)"),
    ("/save [name]", "save this conversation"),
    ("/load [name]", "load a saved conversation (latest if no name)"),
    ("/export", "write this conversation as a markdown file"),
    ("/stats", "show speed, tokens and context usage"),
    ("/config", "show the settings in effect"),
    ("/setup", "run the guided setup again"),
    ("/doctor", "check that everything is installed and running"),
    ("/quit", "leave (Ctrl+D also works; Ctrl+C only stops a reply)"),
]


@dataclass
class App:
    cfg: Config
    paths: Paths
    env: Mapping[str, str]
    ui: UI
    args: argparse.Namespace
    config_error: ConfigError | None = None

    @property
    def interactive(self) -> bool:
        return bool(getattr(self.ui, "interactive", False))


@dataclass
class Stack:
    client: OllamaClient
    agent: Agent
    pipeline: TaskPipeline
    governor: Governor
    store: LogStore
    workspace: Workspace


# -- argument parsing ----------------------------------------------------------------------------


def common_options() -> argparse.ArgumentParser:
    """Options accepted before or after the sub-command. Defaults are suppressed so that only
    flags the user really typed override the config file."""
    parent = argparse.ArgumentParser(add_help=False)
    group = parent.add_argument_group("settings (these override config.toml and environment)")
    sup = argparse.SUPPRESS
    group.add_argument("--config", metavar="FILE", default=sup, help="use this config.toml")
    group.add_argument("--base-url", default=sup, help="Ollama URL, e.g. http://localhost:11434")
    group.add_argument("--model", default=sup, help="Ollama model name, e.g. qwen3:30b")
    group.add_argument("--num-ctx", type=int, default=sup, help="context window in tokens")
    group.add_argument("--mode", choices=list(MODES), default=sup, help="starting mode")
    group.add_argument(
        "--think",
        action=argparse.BooleanOptionalAction,
        default=sup,
        help="force thinking on or off",
    )
    group.add_argument("--workspace", metavar="DIR", default=sup, help="folder the file tools use")
    group.add_argument(
        "--yes",
        "-y",
        action="store_true",
        default=sup,
        help="do not ask before writing files or running code",
    )
    group.add_argument(
        "--govern",
        action="store_true",
        default=sup,
        help="enable the optional Claude governor (needs ANTHROPIC_API_KEY)",
    )
    group.add_argument(
        "--no-verify",
        action="store_true",
        default=sup,
        help="code mode: do not run and repair the model's code",
    )
    group.add_argument(
        "--git",
        action="store_true",
        default=sup,
        help="build mode: git init and commit at checkpoints (off by default)",
    )
    group.add_argument("--plain", action="store_true", default=sup, help="plain text, no colors")
    group.add_argument("--debug", action="store_true", default=sup, help="show tracebacks")
    return parent


def build_parser() -> argparse.ArgumentParser:
    common = common_options()
    parser = argparse.ArgumentParser(
        prog="ai-neo",
        description="AI Neo: a local-first assistant running on Ollama. "
        "With no sub-command it starts the chat.",
        parents=[common],
    )
    parser.add_argument("--version", action="version", version=f"ai-neo {__version__}")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("chat", parents=[common], help="interactive chat (the default)")
    ask = sub.add_parser("ask", parents=[common], help="ask one question and exit")
    ask.add_argument("prompt", nargs="*", help="the question (or pipe it on stdin)")
    ask.add_argument("--tests", metavar="FILE", help="code mode: a test file for solution.py")
    ask.add_argument("--no-tools", action="store_true", help="do not let the model call tools")
    setup = sub.add_parser("setup", parents=[common], help="guided setup wizard")
    setup.add_argument("--non-interactive", action="store_true", help="defaults only, no prompts")
    setup.add_argument("--reset", action="store_true", help="forget which steps were finished")
    sub.add_parser("doctor", parents=[common], help="health check (PASS/WARN/FAIL)")
    evalp = sub.add_parser("eval", parents=[common], help="run an evaluation suite")
    evalp.add_argument("--suite", default="evals", metavar="DIR", help="folder of task files")
    evalp.add_argument("--report", metavar="FILE", help="write the JSON report here")
    evalp.add_argument("--baseline", metavar="FILE", help="earlier report to compare against")
    evalp.add_argument(
        "--target",
        type=float,
        default=DEFAULT_TARGET,
        help="goal for our pass rate as a fraction of the baseline (default 0.9)",
    )
    evalp.add_argument("--category", action="append", help="only this category (repeatable)")
    evalp.add_argument("--repair", type=int, metavar="N", help="repair attempts for Python tasks")
    evalp.add_argument("--no-log", action="store_true", help="do not write task logs")
    export = sub.add_parser(
        "export-dataset", parents=[common], help="export verified, local-only outputs as JSONL"
    )
    export.add_argument("--out", metavar="FILE", help="output file")
    export.add_argument(
        "--include-governed",
        action="store_true",
        help="also include tasks revised after a Claude critique",
    )
    sub.add_parser("config", parents=[common], help="print the settings in effect")
    return parser


FLAG_KEYS = {
    "base_url": "ollama.base_url",
    "model": "ollama.model",
    "num_ctx": "ollama.num_ctx",
    "mode": "chat.mode",
    "workspace": "workspace.path",
}


def overrides_from(args: argparse.Namespace) -> dict[str, object]:
    found: dict[str, object] = {
        key: getattr(args, name) for name, key in FLAG_KEYS.items() if hasattr(args, name)
    }
    if hasattr(args, "think"):
        found["chat.think"] = "on" if args.think else "off"
    return found


# -- wiring --------------------------------------------------------------------------------------


def make_confirmer(ui: UI) -> Callable[[str, str], bool]:
    def confirm(title: str, preview: str) -> bool:
        if not getattr(ui, "interactive", False):
            ui.tag("WARN", f"{title}: skipped because there is no terminal to ask you (use --yes).")
            return False
        ui.preview(f"The model wants to use: {title}", preview)
        return ui.confirm("Allow this?", default=False)

    return confirm


def build_stack(
    app: App,
    *,
    workspace: Workspace | None = None,
    auto_yes: bool | None = None,
) -> Stack:
    cfg, ui = app.cfg, app.ui
    yes = bool(getattr(app.args, "yes", False)) if auto_yes is None else auto_yes
    client = OllamaClient(
        cfg.ollama.base_url,
        cfg.ollama.model,
        keep_alive=cfg.ollama.keep_alive,
        timeout=cfg.ollama.request_timeout,
    )
    space = workspace or Workspace(cfg.workspace)
    registry = ToolRegistry(space, cfg.tools, confirm=make_confirmer(ui), auto_yes=yes)
    agent = Agent(client, registry, cfg, ui)
    governor = Governor(cfg.governor, app.env, ui, assume_yes=yes)
    store = LogStore(app.paths.logs)
    pipeline = TaskPipeline(cfg, agent, space, ui, store=store, governor=governor)
    return Stack(client, agent, pipeline, governor, store, space)


def check_ollama(app: App, stack: Stack) -> bool:
    """Startup health check with friendly, actionable errors."""
    try:
        stack.client.health()
    except (OllamaUnavailable, ModelNotFound) as exc:
        app.ui.tag("FAIL", str(exc))
        app.ui.say("Run `ai-neo doctor` for a full check, or `ai-neo setup` for guided help.")
        return False
    except OllamaError as exc:
        app.ui.tag("FAIL", str(exc))
        return False
    return True


def ensure_workspace(app: App, stack: Stack) -> None:
    if not stack.workspace.root.exists():
        stack.workspace.root.mkdir(parents=True, exist_ok=True)
        app.ui.say(f"Created your workspace folder: {stack.workspace.root}", dim=True)


def maybe_enable_governor(app: App, stack: Stack) -> None:
    if not getattr(app.args, "govern", False):
        return
    reason = stack.governor.enable()
    if reason:
        app.ui.tag("WARN", f"--govern was ignored: {reason}.")
    else:
        app.ui.tag(
            "OK",
            "Claude governor is ON for this session (it costs money; you will be "
            "shown exactly what is sent before the first call).",
        )


# -- the REPL ------------------------------------------------------------------------------------


class Repl:
    def __init__(self, app: App, stack: Stack) -> None:
        self.app = app
        self.stack = stack
        self.ui = app.ui
        self.cfg = app.cfg
        self.sessions = SessionStore(app.paths.sessions)
        self.session = Session(mode=app.cfg.chat.mode, model=stack.client.model)
        self.think: bool | None = None
        self.tools_on = True
        self.verify_on = not bool(getattr(app.args, "no_verify", False))
        self.commands: dict[str, Callable[[str], None]] = {
            "help": self.cmd_help,
            "new": self.cmd_new,
            "think": self.cmd_think,
            "mode": self.cmd_mode,
            "model": self.cmd_model,
            "system": self.cmd_system,
            "tools": self.cmd_tools,
            "govern": self.cmd_govern,
            "save": self.cmd_save,
            "load": self.cmd_load,
            "export": self.cmd_export,
            "stats": self.cmd_stats,
            "config": self.cmd_config,
            "setup": self.cmd_setup,
            "doctor": self.cmd_doctor,
        }

    # -- helpers ---------------------------------------------------------------------------

    def effective_think(self) -> bool:
        return resolve_think(self.cfg.chat.think, self.think, MODES[self.session.mode])

    def prompt_text(self) -> str:
        return f"[{self.session.mode}{'+think' if self.effective_think() else ''}] > "

    def banner(self) -> None:
        cfg = self.cfg
        preset = MODES[self.session.mode]
        self.ui.heading("AI Neo")
        self.ui.say(f"Model: {cfg.ollama.model}  |  Ollama: {cfg.ollama.base_url}")
        self.ui.say(
            f"Mode: {preset.name} ({preset.summary})  |  thinking: "
            f"{'on' if self.effective_think() else 'off'}  |  workspace: {cfg.workspace}"
        )
        self.ui.say(
            "Type your message. /help lists commands. Ctrl+C stops a reply; /quit exits.", dim=True
        )

    def read_message(self) -> str:
        line = self.ui.read_line(self.prompt_text())
        stripped = line.strip()
        if len(stripped) >= 6 and stripped.startswith('"""') and stripped.endswith('"""'):
            return stripped[3:-3].strip()
        if stripped.startswith('"""'):
            lines = [stripped[3:]]
            while True:
                more = self.ui.read_line("... ")
                if more.rstrip().endswith('"""'):
                    lines.append(more.rstrip()[:-3])
                    break
                lines.append(more)
            return "\n".join(lines).strip()
        return line

    # -- main loop -------------------------------------------------------------------------

    def run(self) -> int:
        self.banner()
        while True:
            try:
                line = self.read_message().strip()
                if not line:
                    continue
                if line.startswith("/"):
                    if not self.handle_command(line):
                        return EXIT_OK
                else:
                    self.ask_model(line)
            except EOFError:
                self.ui.say()
                return EXIT_OK
            except KeyboardInterrupt:  # Ctrl+C never leaves the program; /quit or Ctrl+D does
                self.ui.say("\n(Ctrl+C stops a reply. Use /quit or Ctrl+D to leave.)", dim=True)

    def handle_command(self, line: str) -> bool:
        """Run a slash command. Returns False when the user asked to quit."""
        name, _, rest = line[1:].partition(" ")
        name = name.lower()
        if name in ("quit", "exit", "q"):
            return False
        handler = self.commands.get(name)
        if handler is None:
            self.ui.say(f"Unknown command /{name}. Type /help for the list.")
            return True
        try:
            handler(rest.strip())
        except KeyboardInterrupt:
            self.ui.say("(cancelled)", dim=True)
        except EOFError:
            return False
        return True

    def ask_model(self, text: str) -> TaskOutcome | None:
        opts = TaskOptions(
            think=self.think,
            use_tools=self.tools_on,
            verify=self.verify_on,
            git=bool(getattr(self.app.args, "git", False)),
        )
        try:
            outcome = self.stack.pipeline.run(self.session, text, opts)
        except KeyboardInterrupt:
            self.ui.say("\n[stopped]", dim=True)
            return None
        except OllamaError as exc:
            self.ui.tag("FAIL", str(exc))
            return None
        if outcome.interrupted:
            self.ui.say("[stopped by you; the partial answer was kept]", dim=True)
        manager = ContextManager(
            self.cfg.ollama.num_ctx, self.cfg.context.summarize_at, self.cfg.context.keep_last_turns
        )
        extra = f"context {manager.percent(self.session)}% of {self.cfg.ollama.num_ctx}"
        if outcome.stats.eval_count:
            self.ui.stats_line(outcome.stats, extra)
        if outcome.governor and outcome.governor.used:
            last = outcome.governor.rounds[-1].critique.score
            self.ui.say(
                f"[governor: {len(outcome.governor.rounds)} round(s), last score "
                f"{last:g}/10, {outcome.governor.calls} Claude call(s) so far]",
                dim=True,
            )
        return outcome

    # -- commands --------------------------------------------------------------------------

    def cmd_help(self, _: str) -> None:
        self.ui.table(["command", "what it does"], [list(row) for row in HELP_ROWS])
        self.ui.say('Tip: start a message with """ to paste several lines, end with """.', dim=True)

    def cmd_new(self, _: str) -> None:
        self.session = Session(mode=self.session.mode, model=self.stack.client.model)
        self.ui.tag("OK", "Started a new conversation.")

    def cmd_think(self, arg: str) -> None:
        choice = arg.lower()
        if choice not in ("on", "off", "auto"):
            self.ui.say("Usage: /think on|off|auto")
            return
        self.think = {"on": True, "off": False, "auto": None}[choice]
        state = "on" if self.effective_think() else "off"
        self.ui.tag(
            "OK",
            f"Thinking is {state}" + (" (the default for this mode)" if choice == "auto" else "."),
        )
        if self.effective_think():
            self.ui.say(
                "Thinking is slower (often 1-2 minutes on this hardware) but more careful.",
                dim=True,
            )

    def cmd_mode(self, arg: str) -> None:
        if arg not in MODES:
            self.ui.say(f"Usage: /mode {'|'.join(MODES)}")
            return
        self.session.mode = arg
        preset = MODES[arg]
        tools = ", ".join(preset.tools) or "none"
        self.ui.tag("OK", f"Mode: {arg} - {preset.summary}")
        self.ui.say(
            f"Thinking by default: {'on' if preset.think_default else 'off'}; tools: {tools}",
            dim=True,
        )

    def cmd_model(self, arg: str) -> None:
        if not arg:
            self.ui.say(f"Current model: {self.stack.client.model}. Usage: /model <name>")
            return
        try:
            installed = self.stack.client.list_models()
        except OllamaError as exc:
            self.ui.tag("FAIL", str(exc))
            return
        from ai_neo.ollama_client import model_installed

        if not model_installed(installed, arg):
            self.ui.tag("FAIL", missing_model_message(arg))
            self.ui.say("Installed: " + (", ".join(installed) or "(nothing)"), dim=True)
            return
        self.stack.client.model = arg
        self.session.model = arg
        self.ui.tag("OK", f"Now using {arg}. (The first reply is slow while it loads.)")

    def cmd_system(self, arg: str) -> None:
        if not arg:
            self.ui.preview("Current system prompt", self.session.system_prompt())
        elif arg.lower() == "reset":
            self.session.system_override = None
            self.ui.tag("OK", "Back to the built-in prompt for this mode.")
        else:
            self.session.system_override = arg
            self.ui.tag("OK", "Custom system prompt set for this conversation.")

    def cmd_tools(self, arg: str) -> None:
        if arg.lower() not in ("on", "off"):
            self.ui.say("Usage: /tools on|off")
            return
        self.tools_on = arg.lower() == "on"
        self.ui.tag("OK", f"Tools are {'on' if self.tools_on else 'off'}.")

    def cmd_govern(self, arg: str) -> None:
        gov = self.stack.governor
        if arg.lower() == "on":
            reason = gov.enable()
            if reason:
                self.ui.tag("WARN", f"The governor cannot be turned on: {reason}.")
            else:
                self.ui.tag(
                    "OK",
                    "Governor on. It sends your prompt, the draft and tool results to "
                    "Anthropic and costs money. You will see exactly what is sent first.",
                )
        elif arg.lower() == "off":
            gov.disable()
            self.ui.tag("OK", "Governor off.")
        else:
            self.ui.say(f"Governor is {'on' if gov.enabled else 'off'}. Usage: /govern on|off")

    def cmd_save(self, arg: str) -> None:
        path = self.sessions.save(self.session, arg or None)
        self.ui.tag("OK", f"Saved to {path}")

    def cmd_load(self, arg: str) -> None:
        try:
            if not arg:
                known = self.sessions.list()
                if known:
                    self.ui.table(
                        ["name", "mode", "messages"],
                        [[n, m, str(c)] for n, m, c in known],
                        title="Saved conversations (loading the newest)",
                    )
            loaded = self.sessions.load(arg or None)
        except SessionNotFound as exc:
            self.ui.tag("FAIL", str(exc))
            return
        self.session = loaded
        self.ui.tag(
            "OK",
            f"Loaded '{loaded.name or loaded.id}' ({len(loaded.messages)} messages, "
            f"mode {loaded.mode}).",
        )

    def cmd_export(self, _: str) -> None:
        name = safe_name(self.session.name or f"session-{self.session.id}")
        path = self.app.paths.exports / f"{name}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(export_markdown(self.session), encoding="utf-8")
        self.ui.tag("OK", f"Exported to {path}")

    def cmd_stats(self, _: str) -> None:
        s = self.session
        rate = s.generated_tokens / s.generation_seconds if s.generation_seconds else 0.0
        manager = ContextManager(
            self.cfg.ollama.num_ctx, self.cfg.context.summarize_at, self.cfg.context.keep_last_turns
        )
        gov = self.stack.governor
        rows = [
            ["model", self.stack.client.model],
            ["mode", s.mode],
            ["thinking", "on" if self.effective_think() else "off"],
            ["turns", str(s.turns)],
            ["tokens generated", str(s.generated_tokens)],
            ["average speed", f"{rate:.1f} tok/s"],
            [
                "context used",
                f"~{manager.used(s)} of {self.cfg.ollama.num_ctx} tokens ({manager.percent(s)}%)",
            ],
            ["older turns summarized", "yes" if s.summary else "no"],
            [
                "governor",
                f"{'on' if gov.enabled else 'off'} ({gov.calls} calls, {gov.tokens} tokens)",
            ],
        ]
        self.ui.table(["", "value"], rows, title="Session statistics")

    def cmd_config(self, _: str) -> None:
        path = self.app.paths.config
        self.ui.say(f"Config file: {path} ({'found' if path.exists() else 'not created yet'})")
        self.ui.say("Effective settings (after environment variables and flags):", dim=True)
        self.ui.say(render_toml(self.cfg))

    def cmd_setup(self, _: str) -> None:
        from ai_neo.setup.wizard import run_setup

        run_setup(self.app.paths, self.app.env, self.ui, self.cfg, non_interactive=False)
        self.ui.say("Restart ai-neo to use any changed settings.", dim=True)

    def cmd_doctor(self, _: str) -> None:
        from ai_neo.setup.doctor import run_doctor

        run_doctor(self.app.paths, self.app.env, self.ui, self.cfg, self.app.config_error)


# -- sub-commands --------------------------------------------------------------------------------


def offer_first_run_setup(app: App) -> None:
    if app.paths.config.exists() or hasattr(app.args, "config") or not app.interactive:
        return
    app.ui.say("No configuration file exists yet.")
    if app.ui.confirm("Start the guided setup now? (recommended the first time)", default=True):
        from ai_neo.setup.wizard import run_setup

        run_setup(app.paths, app.env, app.ui, app.cfg, non_interactive=False)
        reloaded = reload_config(app)
        if reloaded is not None:
            app.cfg = reloaded
    else:
        app.ui.say("Okay. Run `ai-neo setup` whenever you like.", dim=True)


def reload_config(app: App) -> Config | None:
    try:
        return load_config(app.paths.config, app.env, overrides_from(app.args))
    except ConfigError:
        return None


def cmd_chat(app: App) -> int:
    offer_first_run_setup(app)
    stack = build_stack(app)
    if not check_ollama(app, stack):
        return EXIT_FAIL
    ensure_workspace(app, stack)
    maybe_enable_governor(app, stack)
    repl = Repl(app, stack)
    try:
        return repl.run()
    finally:
        stack.client.close()


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return " ".join(args.prompt).strip()
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    return ""


def cmd_ask(app: App) -> int:
    args = app.args
    prompt = read_prompt(args)
    if not prompt:
        app.ui.say('Usage: ai-neo ask "your question" [--mode code]')
        return EXIT_USAGE
    tests = None
    if args.tests:
        try:
            tests = Path(args.tests).read_text(encoding="utf-8")
        except OSError as exc:
            app.ui.tag("FAIL", f"Cannot read the tests file: {exc}")
            return EXIT_USAGE
    stack = build_stack(app)
    try:
        if not check_ollama(app, stack):
            return EXIT_FAIL
        ensure_workspace(app, stack)
        maybe_enable_governor(app, stack)
        mode = app.cfg.chat.mode
        if mode == "build" and not (getattr(args, "yes", False) or app.interactive):
            app.ui.tag(
                "FAIL",
                "Build mode needs your approval of the plan; run it in a terminal or add --yes.",
            )
            return EXIT_USAGE
        session = Session(mode=mode, model=app.cfg.ollama.model)
        opts = TaskOptions(
            think=getattr(args, "think", None),
            use_tools=not args.no_tools,
            verify=not getattr(args, "no_verify", False),
            tests=tests,
            auto_approve_plan=bool(getattr(args, "yes", False)),
            git=bool(getattr(args, "git", False)),
        )
        outcome = stack.pipeline.run(session, prompt, opts)
        if outcome.stats.eval_count:
            app.ui.stats_line(outcome.stats)
        return exit_code_for(outcome)
    except KeyboardInterrupt:
        app.ui.say("\n[stopped]", dim=True)
        return EXIT_INTERRUPTED
    except OllamaError as exc:
        app.ui.tag("FAIL", str(exc))
        return EXIT_FAIL
    finally:
        stack.client.close()


def exit_code_for(outcome: TaskOutcome) -> int:
    if outcome.error or outcome.interrupted:
        return EXIT_FAIL
    if outcome.verify is not None and not outcome.verify.passed:
        return EXIT_FAIL
    if outcome.build is not None and not outcome.build.passed:
        return EXIT_FAIL
    return EXIT_OK


def cmd_eval(app: App) -> int:
    args = app.args
    try:
        tasks = load_suite(Path(args.suite), args.category)
    except SuiteError as exc:
        app.ui.tag("FAIL", str(exc))
        return EXIT_USAGE
    baseline = None
    if args.baseline:
        try:
            baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            app.ui.tag("FAIL", f"Cannot read the baseline report: {exc}")
            return EXIT_USAGE
    with tempfile.TemporaryDirectory(prefix="ai_neo_eval_", ignore_cleanup_errors=True) as scratch:
        stack = build_stack(app, workspace=Workspace(Path(scratch)), auto_yes=True)
        try:
            if not check_ollama(app, stack):
                return EXIT_FAIL
            maybe_enable_governor(app, stack)
            app.ui.say(f"Running {len(tasks)} task(s). Build tasks work in a temporary folder.")
            store = None if args.no_log else stack.store
            runner = PipelineRunner(stack.pipeline, repair_attempts=args.repair)
            report = run_suite(
                tasks, runner, app.cfg, ui=app.ui, store=store, suite_name=str(args.suite)
            )
        except KeyboardInterrupt:
            app.ui.say("\n[evaluation stopped]", dim=True)
            return EXIT_INTERRUPTED
        except OllamaError as exc:
            app.ui.tag("FAIL", str(exc))
            return EXIT_FAIL
        finally:
            stack.client.close()
    data = report.to_dict(baseline, args.target)
    render_report(data, app.ui)
    if args.report:
        write_report(data, Path(args.report))
        app.ui.tag("OK", f"Report written to {args.report}")
    return EXIT_OK


def cmd_export_dataset(app: App) -> int:
    args = app.args
    out = (
        Path(args.out)
        if args.out
        else (app.paths.datasets / f"dataset-{datetime.now():%Y%m%d-%H%M%S}.jsonl")
    )
    stats = LogStore(app.paths.logs).export_dataset(out, include_governed=args.include_governed)
    app.ui.tag("OK", f"Wrote {stats.exported} example(s) to {out}")
    app.ui.say(
        f"Looked at {stats.total} logged task(s): {stats.skipped_unverified} skipped "
        f"(not verified by tests or an exact match), {stats.skipped_governed} skipped "
        "(revised after a Claude critique)."
    )
    app.ui.say(
        "Only the local model's own outputs are exported; Claude's critique text is never "
        "included.",
        dim=True,
    )
    if args.include_governed:
        app.ui.tag(
            "WARN",
            "You included tasks that were revised using Claude's feedback. Before "
            "using any Claude-influenced text for training, check Anthropic's current terms "
            "of use.",
        )
    return EXIT_OK


def cmd_setup(app: App) -> int:
    from ai_neo.setup.wizard import run_setup

    return run_setup(
        app.paths,
        app.env,
        app.ui,
        app.cfg,
        non_interactive=app.args.non_interactive,
        reset=app.args.reset,
    )


def cmd_doctor(app: App) -> int:
    from ai_neo.setup.doctor import run_doctor

    return run_doctor(app.paths, app.env, app.ui, app.cfg, app.config_error)


def cmd_config(app: App) -> int:
    app.ui.say(
        f"Config file: {app.paths.config} "
        f"({'found' if app.paths.config.exists() else 'not created yet'})"
    )
    app.ui.say(render_toml(app.cfg))
    return EXIT_OK


COMMANDS: dict[str, Callable[[App], int]] = {
    "chat": cmd_chat,
    "ask": cmd_ask,
    "setup": cmd_setup,
    "doctor": cmd_doctor,
    "eval": cmd_eval,
    "export-dataset": cmd_export_dataset,
    "config": cmd_config,
}


# -- entry point ---------------------------------------------------------------------------------


def prepare_streams() -> None:
    """Never crash on characters the console cannot show (common on Windows)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(OSError, ValueError):
                reconfigure(encoding="utf-8", errors="replace")


def load_app(args: argparse.Namespace, env: Mapping[str, str], ui: UI) -> App:
    paths = Paths.default(env)
    config_path = Path(args.config).expanduser() if hasattr(args, "config") else paths.config
    error: ConfigError | None = None
    try:
        cfg = load_config(config_path, env, overrides_from(args))
    except ConfigError as exc:
        error = exc
        cfg = default_config()
    return App(cfg, paths, env, ui, args, error)


def main(argv: Sequence[str] | None = None) -> int:
    prepare_streams()
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    command = args.command or "chat"
    env = os.environ
    ui = make_ui(plain=bool(getattr(args, "plain", False)))
    app = load_app(args, env, ui)
    if app.config_error is not None and command not in ("doctor", "setup"):
        ui.tag("FAIL", "The configuration has problems:")
        for problem in app.config_error.problems:
            ui.say(f"  - {problem}")
        ui.say(f"Fix them in {app.paths.config} (or delete it and run `ai-neo setup`).")
        return EXIT_USAGE
    try:
        return COMMANDS[command](app)
    except KeyboardInterrupt:
        ui.say("\n[interrupted]")
        return EXIT_INTERRUPTED
    except EOFError:
        ui.say("\n[input ended]")
        return EXIT_FAIL
    except Exception as exc:  # last resort: no raw tracebacks for beginners
        if getattr(args, "debug", False) or env.get("AI_NEO_DEBUG"):
            raise
        ui.tag("FAIL", f"Unexpected problem: {type(exc).__name__}: {exc}")
        ui.say("Run again with --debug for technical details, and `ai-neo doctor` to check setup.")
        return EXIT_FAIL


__all__ = ["main", "build_parser", "Repl"]
