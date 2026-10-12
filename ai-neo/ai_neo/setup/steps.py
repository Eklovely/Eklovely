"""The nine steps of the guided setup, in order."""

from __future__ import annotations

import importlib.util
from typing import Any

from ai_neo.config import (
    ConfigError,
    config_values,
    default_config,
    is_local_url,
    load_config,
    write_config,
)
from ai_neo.governor import Governor, key_present
from ai_neo.ollama_client import (
    ModelNotFound,
    OllamaError,
    OllamaUnavailable,
    PullCancelled,
    PullFailed,
    model_installed,
)
from ai_neo.prompts import SELF_TEST_ECHO, THINKING
from ai_neo.setup.checks import (
    MODEL_DISK_GB,
    Status,
    check_disk,
    check_python,
    check_ram,
    gpu_check,
    worst,
)
from ai_neo.setup.probe import detect_gpu, ollama_models_dir
from ai_neo.setup.selftest import run_selftest
from ai_neo.setup.wizard import SetupContext, Step, StepResult
from ai_neo.types import Message, Sampling

OLLAMA_DOWNLOAD_URL = "https://ollama.com/download"
MODEL_SIZES = {"qwen3:30b": f"about {MODEL_DISK_GB} GB"}


def _remote(ctx: SetupContext) -> bool:
    return not is_local_url(ctx.cfg.ollama.base_url)


def _show(ctx: SetupContext, status: Status, text: str, fix: str = "") -> None:
    ctx.ui.tag(status.value, text)
    if fix and status in (Status.WARN, Status.FAIL):
        ctx.ui.say(f"      -> {fix}")


# -- 1. system check -----------------------------------------------------------------------------


def preview_system(ctx: SetupContext) -> list[str]:
    actions = ["Read your operating system, Python version, memory and free disk space"]
    if not _remote(ctx):
        actions.append(
            "Run (read-only, only if you allow it): nvidia-smi --query-gpu=name,memory.total "
            "--format=csv,noheader,nounits"
        )
    return actions


def run_system(ctx: SetupContext) -> StepResult:
    ui, probe = ctx.ui, ctx.probe
    ui.say(
        "Requirements: Python 3.11+; qwen3:30b needs about 19 GB of disk space and 32 GB of RAM "
        "is comfortable (24 GB is the minimum I recommend)."
    )
    ui.tag("OK", f"Operating system: {probe.os_name()}")
    checks = [check_python(probe)]
    gpu = None
    if _remote(ctx):
        ui.tag(
            "SKIP",
            f"Memory, disk and GPU checks skipped: Ollama runs on another machine "
            f"({ctx.cfg.ollama.base_url}).",
        )
    else:
        checks.append(check_ram(probe))
        checks.append(check_disk(probe, ollama_models_dir(probe)))
        if ctx.consent(
            "May I run the read-only `nvidia-smi` command to look for a GPU?", kind="probe"
        ):
            gpu = detect_gpu(probe)
            checks.append(gpu_check(gpu))
        else:
            ui.say("GPU check skipped at your request.", dim=True)
    ctx.facts["ram_gb"] = probe.total_ram_gb()
    ctx.facts["gpu"] = gpu
    for check in checks:
        _show(ctx, check.status, f"{check.label}: {check.detail}", check.fix)
    if not _remote(ctx):
        slow = gpu is None or (gpu.vram_mb is not None and gpu.vram_mb < 12000)
        if slow:
            ui.say(
                "What to expect: most of the model will run on the CPU, so answers are slower "
                "(roughly 8-15 words per second on a laptop like yours) but they work."
            )
    status = worst([c.status for c in checks])
    summary = {
        Status.OK: "This computer meets the requirements.",
        Status.WARN: "This computer can probably run it, with the caveats above.",
        Status.FAIL: "Something blocks AI Neo; see the fix above.",
    }[status]
    return StepResult(status, summary)


# -- 2. Ollama installed -------------------------------------------------------------------------


def preview_installed(ctx: SetupContext) -> list[str]:
    if _remote(ctx):
        return ["Nothing: you use a remote Ollama server, so no local install is needed"]
    return [
        "Look for `ollama` on your PATH and in the usual install folder "
        "(%LOCALAPPDATA%\\Programs\\Ollama on Windows)",
        "Run `ollama --version` (read-only, only if you allow it)",
    ]


def run_installed(ctx: SetupContext) -> StepResult:
    ui, probe = ctx.ui, ctx.probe
    if _remote(ctx):
        return StepResult(Status.SKIP, f"Skipped: Ollama is expected at {ctx.cfg.ollama.base_url}.")
    found = probe.find_executable("ollama")
    if found is None and not ctx.non_interactive:
        ui.say("Ollama is not installed (or I cannot find it). To install it:")
        ui.say(f"  1. Open {OLLAMA_DOWNLOAD_URL} in your browser")
        ui.say(
            "  2. Download the installer for your system and run it (I will not do that for you)"
        )
        ui.say("  3. When it finishes, come back to this window")
        ui.read_line("Press Enter once Ollama is installed... ")
        found = probe.find_executable("ollama")
    if found is None:
        return StepResult(
            Status.FAIL, f"Ollama was not found. Download it from {OLLAMA_DOWNLOAD_URL}."
        )
    version = ""
    if ctx.consent("May I run `ollama --version` to confirm it works?", kind="probe"):
        done = probe.run([found, "--version"])
        if done.returncode == 0:
            version = " - " + (done.stdout.strip() or done.stderr.strip()).splitlines()[-1][:80]
    return StepResult(Status.OK, f"Found Ollama at {found}{version}")


# -- 3. Ollama running ---------------------------------------------------------------------------


def preview_running(ctx: SetupContext) -> list[str]:
    return [f"GET {ctx.cfg.ollama.base_url}/api/tags  (asks the server which models it has)"]


def run_running(ctx: SetupContext) -> StepResult:
    with ctx.client() as client:
        try:
            models = client.list_models()
        except OllamaUnavailable:
            ctx.server_ok, ctx.models = False, None
            if is_local_url(ctx.cfg.ollama.base_url):
                ctx.ui.say(
                    "To start it: open the Ollama app from the Start menu (it lives in the "
                    "system tray), or run `ollama serve` in a separate terminal window. "
                    "Wait a few seconds, then choose Retry."
                )
            return StepResult(Status.FAIL, f"Nothing answered at {ctx.cfg.ollama.base_url}.")
        except OllamaError as exc:
            ctx.server_ok, ctx.models = False, None
            return StepResult(Status.FAIL, str(exc))
    ctx.server_ok, ctx.models = True, models
    return StepResult(Status.OK, f"Ollama is running ({len(models)} model(s) installed).")


# -- 4. model installed --------------------------------------------------------------------------


def preview_model(ctx: SetupContext) -> list[str]:
    model = ctx.cfg.ollama.model
    return [
        f"Check whether {model} is installed",
        f"If not (with your permission): POST {ctx.cfg.ollama.base_url}/api/pull for {model} "
        f"(same as `ollama pull {model}`)",
    ]


def run_model(ctx: SetupContext) -> StepResult:
    model = ctx.cfg.ollama.model
    ui = ctx.ui
    with ctx.client() as client:
        try:
            models = client.list_models()
        except OllamaError:
            return StepResult(
                Status.FAIL,
                "Ollama is not answering, so I cannot check the models. "
                "Go back to step 3 (choose Quit, start Ollama, run setup again).",
            )
        ctx.models, ctx.server_ok = models, True
        if model_installed(models, model):
            return StepResult(Status.OK, f"{model} is already installed.")
        size = MODEL_SIZES.get(model, "several gigabytes (size unknown)")
        free = ctx.probe.free_disk_gb(ollama_models_dir(ctx.probe))
        free_text = f"; you have about {free:.0f} GB free" if free is not None else ""
        ui.say(
            f"{model} is not installed. The download is {size}{free_text}. It can take a while, "
            "and you can press Ctrl+C to pause: running this step again resumes."
        )
        if not ctx.consent(f"Download {model} now?", kind="download"):
            return StepResult(
                Status.FAIL,
                f"{model} is not installed (nothing was downloaded). "
                f"Install it yourself with: ollama pull {model}",
            )
        bar = ui.progress(f"pulling {model}")
        try:
            client.pull(model, lambda p: bar.update(p.completed, p.total, p.status))
        except PullCancelled as exc:
            return StepResult(Status.FAIL, str(exc))
        except (PullFailed, OllamaError) as exc:
            return StepResult(Status.FAIL, f"The download failed: {exc}")
        finally:
            bar.close()
        try:
            models = client.list_models()
        except OllamaError as exc:
            return StepResult(Status.FAIL, str(exc))
    ctx.models = models
    if model_installed(models, model):
        return StepResult(Status.OK, f"{model} downloaded and installed.")
    return StepResult(Status.FAIL, "The download finished but the model is not listed yet.")


# -- 5. first test -------------------------------------------------------------------------------


def preview_first(ctx: SetupContext) -> list[str]:
    return [
        f"Send a tiny prompt to {ctx.cfg.ollama.model}: {SELF_TEST_ECHO!r}",
        "Stream the reply and measure tokens per second",
    ]


def run_first(ctx: SetupContext) -> StepResult:
    ui = ctx.ui
    cfg = ctx.cfg
    ui.say(
        "The very first request is slow because Ollama has to load the whole model into memory "
        f"(this can take a minute or two). With keep_alive={cfg.ollama.keep_alive} it then "
        "stays loaded, so later requests start quickly."
    )
    ui.say(
        "Watch for two kinds of text: the model's thinking (shown dimmed, or collapsed into a "
        "[thought for ...] note) and then its answer."
    )
    sampling = Sampling(
        THINKING.temperature, THINKING.top_p, THINKING.top_k, cfg.ollama.num_ctx, 400
    )
    renderer = ui.stream()
    with ctx.client() as client:
        try:
            result = client.chat(
                [Message("user", SELF_TEST_ECHO)],
                sampling=sampling,
                think=True,
                on_event=renderer.event,
            )
        except (OllamaUnavailable, ModelNotFound, OllamaError) as exc:
            return StepResult(Status.FAIL, str(exc))
        finally:
            renderer.close()
    if result.interrupted:
        return StepResult(Status.FAIL, "Stopped with Ctrl+C before the model finished.")
    stats = result.stats
    ui.stats_line(stats)
    ctx.facts["tok_per_s"] = stats.tok_per_s
    load_seconds = stats.load_duration_ns / 1e9
    if load_seconds > 1:
        ui.say(f"Loading the model took {load_seconds:.1f}s. Later requests skip that.")
    if not result.content:
        return StepResult(
            Status.WARN,
            "The model produced thinking but no answer within the small "
            "test budget. That is okay; real questions get more room.",
        )
    if stats.tok_per_s and stats.tok_per_s < 3:
        return StepResult(
            Status.WARN,
            f"It works, but at {stats.tok_per_s:.1f} tokens/second it is "
            "very slow. Close other programs, or lower the context length in step 6.",
        )
    return StepResult(
        Status.OK,
        f"The model answered {result.content[:40]!r} at {stats.tok_per_s:.1f} tokens/second.",
    )


# -- 6. choose settings --------------------------------------------------------------------------


def _base_config(ctx: SetupContext) -> Any:
    try:
        return (
            load_config(ctx.paths.config, {}, None)
            if ctx.paths.config.exists()
            else default_config()
        )
    except ConfigError:
        return default_config()


def save_settings(ctx: SetupContext, changes: dict[str, Any]) -> None:
    """Write config.toml = (what is in the file now) + changes, then reload the effective config."""
    merged = {**config_values(_base_config(ctx)), **changes}
    write_config(load_config(None, {}, merged), ctx.paths.config)
    ctx.cfg = load_config(ctx.paths.config, ctx.env)


def _ask_setting(ctx: SetupContext, key: str, question: str, default: str) -> str:
    for _ in range(3):
        answer = ctx.ui.ask(question, default)
        try:
            load_config(None, {}, {key: answer})
        except ConfigError as exc:
            ctx.ui.tag("WARN", exc.problems[0])
            continue
        return answer
    ctx.ui.say(f"Using the recommended value {default}.")
    return default


def preview_settings(ctx: SetupContext) -> list[str]:
    return [
        "Ask four short questions (press Enter to accept each recommended value)",
        f"Write {ctx.paths.config} and create your workspace folder if it is missing",
    ]


def run_settings(ctx: SetupContext) -> StepResult:
    ui = ctx.ui
    current = ctx.cfg
    ram = ctx.facts.get("ram_gb")
    has_file = ctx.paths.config.exists()
    recommended_ctx = "4096" if ram is not None and ram < 24 else "8192"
    default_ctx = str(current.ollama.num_ctx) if has_file else recommended_ctx
    if ctx.non_interactive:
        changes: dict[str, Any] = {
            "workspace.path": str(current.workspace),
            "ollama.num_ctx": default_ctx,
            "chat.think": current.chat.think,
            "ollama.keep_alive": current.ollama.keep_alive,
        }
    else:
        ui.say("Workspace: the only folder where AI Neo may read and write project files.")
        workspace = _ask_setting(ctx, "workspace.path", "Workspace folder", str(current.workspace))
        ui.say(
            "Context length: how much conversation the model can keep in mind. "
            f"{recommended_ctx} is recommended for your computer. Bigger values remember more of "
            "a long chat, but each extra 8192 tokens uses roughly another 1 GB of RAM and slows "
            "things down."
        )
        num_ctx = _ask_setting(ctx, "ollama.num_ctx", "Context length (tokens)", default_ctx)
        ui.say(
            "Thinking: 'on' makes the model reason step by step first - slower but better for "
            "hard questions. 'off' is fastest. 'auto' (recommended) thinks for reasoning, math "
            "and code, and skips it for writing and building."
        )
        think = _ask_setting(
            ctx, "chat.think", "Thinking by default (auto/on/off)", current.chat.think
        )
        ui.say(
            "keep_alive: how long Ollama keeps the model loaded in memory after your last "
            "question. Longer means faster follow-ups but uses RAM while idle."
        )
        keep_alive = _ask_setting(
            ctx, "ollama.keep_alive", "Keep the model loaded for", current.ollama.keep_alive
        )
        changes = {
            "workspace.path": workspace,
            "ollama.num_ctx": num_ctx,
            "chat.think": think,
            "ollama.keep_alive": keep_alive,
        }
    merged = load_config(None, {}, {**config_values(_base_config(ctx)), **changes})
    question = (
        f"Write these settings to {ctx.paths.config} and create {merged.workspace} if needed?"
    )
    if not ctx.consent(question, kind="write"):
        return StepResult(Status.FAIL, "Settings were not saved because you declined.")
    merged.workspace.mkdir(parents=True, exist_ok=True)
    save_settings(ctx, changes)
    ui.table(
        ["setting", "value"],
        [
            ["workspace", str(ctx.cfg.workspace)],
            ["context length", str(ctx.cfg.ollama.num_ctx)],
            ["thinking default", ctx.cfg.chat.think],
            ["keep_alive", ctx.cfg.ollama.keep_alive],
            ["model", ctx.cfg.ollama.model],
        ],
    )
    return StepResult(Status.OK, f"Settings saved to {ctx.paths.config}")


# -- 7. optional governor ------------------------------------------------------------------------


def preview_governor(ctx: SetupContext) -> list[str]:
    return ["Explain the optional Claude governor and ask whether you want it (default: No)"]


def _key_commands(os_name: str) -> list[str]:
    if os_name == "Windows":
        return ['PowerShell or Command Prompt:  setx ANTHROPIC_API_KEY "your-key"']
    return [
        "macOS (zsh):  echo 'export ANTHROPIC_API_KEY=\"your-key\"' >> ~/.zshrc",
        "Linux (bash):  echo 'export ANTHROPIC_API_KEY=\"your-key\"' >> ~/.bashrc",
    ]


def run_governor(ctx: SetupContext) -> StepResult:
    ui = ctx.ui
    ui.say(
        "The governor is OPTIONAL and OFF by default. When you switch it on (--govern or "
        "/govern on), Claude reviews each draft answer from your local model and the local model "
        "improves it. That means your prompt, the draft and any tool results are sent to "
        "Anthropic's servers, and every call costs money. Nothing is sent unless you turn it on, "
        "and you see exactly what is sent before the first call."
    )
    if ctx.non_interactive or not ui.confirm("Do you want to set up the governor?", default=False):
        return StepResult(Status.OK, "Not setting up the governor (the default). It stays off.")
    ui.say(
        "AI Neo never asks you to type your API key. Instead, set it as an environment "
        "variable (replace your-key with your key from your Anthropic console):"
    )
    for command in _key_commands(ctx.probe.os_name()):
        ui.say(f"  {command}")
    ui.say(
        "Then close this terminal window and open a NEW one (the variable only appears in new "
        "windows), and run `ai-neo setup` again - finished steps are skipped."
    )
    if not key_present(ctx.env):
        return StepResult(
            Status.FAIL,
            "ANTHROPIC_API_KEY is not set in this window yet. (I only "
            "checked that it exists; I never read or show its value.)",
        )
    ui.tag("OK", "ANTHROPIC_API_KEY is set (its value is never shown or stored).")
    model = ctx.cfg.governor.claude_model or ui.ask(
        "Which Claude model should the governor use? Copy the name from Anthropic's model list "
        "(there is no built-in default)",
        "",
    )
    if model:
        save_settings(ctx, {"governor.claude_model": model})
    sdk_ready = importlib.util.find_spec("anthropic") is not None
    if not sdk_ready:
        ui.say("The 'anthropic' package is not installed. Run:  pip install -e .[claude]")
    if not model or not sdk_ready:
        return StepResult(
            Status.WARN,
            "The governor is not ready yet (see the notes above); it is "
            "optional, so setup can continue.",
        )
    if ctx.consent(
        "Make ONE tiny test call to Claude now? (it costs a tiny amount)", kind="api", default=False
    ):
        governor = Governor(ctx.cfg.governor, ctx.env, ui)
        if (
            governor.enable() is None
            and governor.review("reasoning", "Reply with: ok", "ok") is not None
        ):
            return StepResult(Status.OK, "The governor works. It stays off until you use --govern.")
        return StepResult(
            Status.WARN,
            "The test call did not succeed (see above). The governor will "
            "fall back to local answers.",
        )
    return StepResult(Status.OK, "The governor is configured. It stays off until you use --govern.")


# -- 8. self-test --------------------------------------------------------------------------------


def preview_selftest(ctx: SetupContext) -> list[str]:
    return [
        "Calculator: 2+2*3",
        "run_python: a one-line script, then `while True: pass`, which the timeout must stop",
        "File tools: a path like ../ and a symlink escape must be rejected",
        "One tiny task per category (reasoning, computation, python, writing, build), offline",
    ]


def run_selftest_step(ctx: SetupContext) -> StepResult:
    ui = ctx.ui
    if not ctx.consent(
        "These checks start a few short Python subprocesses on this computer. Run them?",
        kind="probe",
    ):
        return StepResult(Status.FAIL, "The self-test was not run because you declined.")
    checks = run_selftest(ctx.cfg)
    for check in checks:
        _show(ctx, check.status, f"{check.label}: {check.detail}", check.fix)
    status = worst([c.status for c in checks])
    passed = sum(c.status is Status.OK for c in checks)
    ui.say(
        "Note: these use no model and no network; they test AI Neo's own safety features.", dim=True
    )
    return StepResult(status, f"{passed} of {len(checks)} self-test checks passed.")


# -- 9. done -------------------------------------------------------------------------------------


def preview_done(ctx: SetupContext) -> list[str]:
    return []


def run_done(ctx: SetupContext) -> StepResult:
    ui, cfg, paths = ctx.ui, ctx.cfg, ctx.paths
    speed = ctx.facts.get("tok_per_s")
    gov = (
        "ready (off until --govern)"
        if cfg.governor.claude_model and key_present(ctx.env)
        else "not set up"
    )
    ui.table(
        ["what", "where / value"],
        [
            ["model", f"{cfg.ollama.model} at {cfg.ollama.base_url}"],
            ["settings file", str(paths.config)],
            ["workspace (your projects)", str(cfg.workspace)],
            ["saved chats", str(paths.sessions)],
            ["task logs", str(paths.logs)],
            ["speed measured", f"{speed:.1f} tokens/second" if speed else "not measured"],
            ["Claude governor", gov],
        ],
        title="Your setup",
    )
    ui.say("Try these next:")
    ui.say("  Chat:           ai-neo")
    ui.say(
        '  A Python task:  ai-neo ask "Write a function that reverses the words in a sentence" '
        "--mode code"
    )
    ui.say(
        "  Build a project: ai-neo --mode build   (then describe the project; you approve the plan)"
    )
    ui.say(
        "Re-run `ai-neo setup` any time to change settings, and `ai-neo doctor` to check health."
    )
    if ctx.unresolved:
        ui.tag(
            "WARN",
            "Setup is not complete. Unfinished steps: "
            + ", ".join(ctx.unresolved)
            + ". Run `ai-neo setup` again to finish them.",
        )
        return StepResult(Status.WARN, "Setup finished with unresolved steps.")
    return StepResult(Status.OK, "Setup is complete. Enjoy AI Neo!")


STEPS: tuple[Step, ...] = (
    Step(
        "system",
        "System check",
        "This looks at your computer: operating system, Python, memory, free disk space and GPU.",
        "qwen3:30b is a big model, so we check up front that it will fit and tell you what speed "
        "to expect.",
        (
            "Python too old: install Python 3.11+ from https://www.python.org/downloads/ and tick "
            "'Add python.exe to PATH', then open a new terminal.",
            "Low RAM: close other programs, lower the context length, or use a smaller model.",
            "Low disk space: free about 25 GB, or point OLLAMA_MODELS at a bigger drive.",
            "No GPU found: normal. AI Neo works on the CPU, just slower.",
        ),
        preview_system,
        run_system,
    ),
    Step(
        "ollama-installed",
        "Is Ollama installed?",
        "Ollama is the free program that runs the AI model on your computer.",
        "AI Neo talks to Ollama; without it there is no model to talk to.",
        (
            f"Download it from {OLLAMA_DOWNLOAD_URL}, run the installer, then choose Retry.",
            "Installed but not found? Open a NEW terminal window so PATH refreshes, or reboot.",
            "On Windows it normally lives in %LOCALAPPDATA%\\Programs\\Ollama.",
        ),
        preview_installed,
        run_installed,
    ),
    Step(
        "ollama-running",
        "Is Ollama running?",
        "Ollama must be running in the background, listening on port 11434.",
        "AI Neo sends your questions to that local server.",
        (
            "Start the Ollama app from the Start menu (llama icon in the system tray).",
            "Or run `ollama serve` in a separate terminal and leave it open.",
            "Port 11434 already in use? Another copy of Ollama may be running - that is fine. "
            "Otherwise set OLLAMA_HOST or change ollama.base_url.",
            "A firewall prompt may appear the first time: allow Ollama on private networks.",
            "Using a remote server? Check ollama.base_url and that the machine is reachable.",
        ),
        preview_running,
        run_running,
    ),
    Step(
        "model-installed",
        "Is the model installed?",
        "The model (qwen3:30b by default) is a large file Ollama downloads once.",
        "Without it Ollama has nothing to run.",
        (
            "Not enough disk space: free about 25 GB and Retry.",
            "Slow or failing download: check your internet, VPN or proxy. A proxy must be set for "
            "the Ollama server itself (HTTPS_PROXY), then restart Ollama.",
            "Download stopped halfway: choose Retry - Ollama resumes where it left off.",
            "Antivirus or firewall blocking ollama.com: allow Ollama, then Retry.",
            "You can also download it yourself: ollama pull qwen3:30b",
        ),
        preview_model,
        run_model,
    ),
    Step(
        "first-test",
        "First test",
        "This sends one tiny question to the model and shows the streamed answer.",
        "It proves the whole chain works and measures how fast your computer is.",
        (
            "Out of memory: close other programs, lower the context length, or use a smaller one.",
            "Very slow: normal on CPU for the first request (the model is loading). Retry once.",
            "Timeout: raise ollama.request_timeout in config.toml.",
            "Model not found: go back to the model step and install it.",
        ),
        preview_first,
        run_first,
    ),
    Step(
        "settings",
        "Choose settings",
        "A few simple questions: workspace folder, context length, thinking and keep-alive.",
        "These are saved to config.toml so you never have to repeat them.",
        (
            "Could not write the file: check that the folder is writable, or set AI_NEO_HOME.",
            "Typed something invalid? The message says what is allowed; just answer again.",
            "Prefer editing by hand? Open the config.toml printed after this step.",
        ),
        preview_settings,
        run_settings,
    ),
    Step(
        "governor",
        "Optional: Claude governor",
        "The governor lets Claude (from Anthropic) critique your local model's answers.",
        "It can improve quality, but it is optional, costs money, and sends text to Anthropic.",
        (
            "setx only affects NEW terminal windows: close this one, open a new one, and run "
            "`ai-neo setup` again.",
            "Do not paste your key into AI Neo; set the environment variable as shown.",
            "You can skip this step: everything else works without it.",
        ),
        preview_governor,
        run_governor,
    ),
    Step(
        "self-test",
        "Self-test",
        "A quick offline check of AI Neo's own tools and safety features.",
        "It confirms that code runs are time-limited and that files outside your workspace are "
        "protected.",
        (
            "A failing run_python check: antivirus may block Python subprocesses; allow python.exe",
            "A failing path check is a bug: please report it and do not use the file tools.",
            "Symlink check skipped on Windows is normal without Developer Mode.",
        ),
        preview_selftest,
        run_selftest_step,
    ),
    Step(
        "done",
        "All done",
        "A summary of what is set up and what to try next.",
        "So you know where everything lives.",
        (),
        preview_done,
        run_done,
        always_run=True,
    ),
)
