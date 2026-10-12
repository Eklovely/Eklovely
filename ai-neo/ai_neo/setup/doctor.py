"""`ai-neo doctor`: a non-interactive health check. Exit code 1 if anything FAILs."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from ai_neo.config import Config, ConfigError, is_local_url
from ai_neo.ollama_client import OllamaClient, model_installed
from ai_neo.paths import Paths
from ai_neo.setup.checks import (
    Check,
    Status,
    check_config,
    check_dependencies,
    check_disk,
    check_governor,
    check_gpu,
    check_model,
    check_ollama_binary,
    check_python,
    check_ram,
    check_server,
    check_userdata,
    check_workspace,
)
from ai_neo.setup.probe import RealProbe, SystemProbe, ollama_models_dir
from ai_neo.setup.selftest import run_selftest
from ai_neo.setup.wizard import default_client_factory
from ai_neo.ui import UI

LABELS = {Status.OK: "PASS", Status.WARN: "WARN", Status.FAIL: "FAIL", Status.SKIP: "SKIP"}


SelfTest = Callable[[Config], list[Check]]


def selftest_check(cfg: Config, selftest: SelfTest) -> Check:
    """One doctor line summarizing the offline self-test of the tools and safety guards."""
    results = selftest(cfg)
    failed = [c for c in results if c.status is Status.FAIL]
    if failed:
        first = failed[0]
        return Check(
            "selftest",
            "Tools and safety self-test",
            Status.FAIL,
            f"{len(failed)} of {len(results)} checks failed (first: {first.label})",
            first.fix or "Run `ai-neo setup` for details.",
        )
    skipped = [c for c in results if c.status is Status.WARN]
    if skipped:
        return Check(
            "selftest",
            "Tools and safety self-test",
            Status.WARN,
            f"{len(results) - len(skipped)} of {len(results)} checks passed; "
            f"{len(skipped)} skipped ({skipped[0].label})",
            skipped[0].fix,
        )
    return Check(
        "selftest",
        "Tools and safety self-test",
        Status.OK,
        f"all {len(results)} checks passed (calculator, code timeout, path guard, checkers)",
    )


def collect_checks(
    paths: Paths,
    env: Mapping[str, str],
    cfg: Config,
    config_error: ConfigError | None,
    probe: SystemProbe,
    client: OllamaClient,
    selftest: SelfTest = run_selftest,
) -> list[Check]:
    local = is_local_url(cfg.ollama.base_url)
    server, models = check_server(client)
    installed = models is not None and model_installed(models, cfg.ollama.model)
    checks = [
        check_python(probe),
        check_dependencies(),
        check_config(paths.config, config_error),
        check_userdata(paths),
    ]
    if local:
        checks.append(check_ram(probe))
        checks.append(check_disk(probe, ollama_models_dir(probe), 0 if installed else 25))
        checks.append(check_gpu(probe))
        checks.append(check_ollama_binary(probe, server.status is Status.OK))
    else:
        checks.append(
            Check(
                "remote",
                "Local hardware and install",
                Status.SKIP,
                f"Ollama runs on another machine ({cfg.ollama.base_url})",
            )
        )
    checks += [
        server,
        check_model(models, cfg.ollama.model),
        check_workspace(cfg),
        check_governor(cfg, env),
        selftest_check(cfg, selftest),
    ]
    return checks


def run_doctor(
    paths: Paths,
    env: Mapping[str, str],
    ui: UI,
    cfg: Config,
    config_error: ConfigError | None = None,
    *,
    probe: SystemProbe | None = None,
    client_factory: Callable[[Config], OllamaClient] | None = None,
    selftest: SelfTest = run_selftest,
) -> int:
    ui.heading("AI Neo doctor")
    with (client_factory or default_client_factory)(cfg) as client:
        checks = collect_checks(
            paths, env, cfg, config_error, probe or RealProbe(), client, selftest
        )
    for check in checks:
        ui.tag(LABELS[check.status], f"{check.label}: {check.detail}")
        if check.fix and check.status in (Status.WARN, Status.FAIL):
            ui.say(f"       fix: {check.fix}")
    counts = {status: sum(c.status is status for c in checks) for status in Status}
    ui.say()
    ui.say(
        f"{counts[Status.OK]} passed, {counts[Status.WARN]} warning(s), "
        f"{counts[Status.FAIL]} failed."
    )
    if counts[Status.FAIL]:
        ui.say("Fix the FAIL items above (or run `ai-neo setup` for guided help).")
        return 1
    return 0
