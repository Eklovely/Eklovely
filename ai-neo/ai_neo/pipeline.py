"""One task from prompt to logged result: mode routing, verification, governor and logging.

The interactive CLI, `ai-neo ask` and the evaluation harness all go through `TaskPipeline.run`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ai_neo.agent import Agent, ToolEvent, TurnResult, resolve_think, sampling_for
from ai_neo.build import Builder, BuildReport
from ai_neo.config import Config
from ai_neo.governor import Governor, GovernResult, govern
from ai_neo.logging_store import LogStore, TaskRecord
from ai_neo.prompts import MODES
from ai_neo.sessions import Session
from ai_neo.textutil import clip_middle, extract_python
from ai_neo.tools import Workspace
from ai_neo.types import GenStats, Message
from ai_neo.ui import UI
from ai_neo.verify import (
    ComputeReport,
    VerifyReport,
    run_compute_task,
    run_python_task,
    run_solution,
    verify_and_repair,
)


@dataclass
class TaskOptions:
    think: bool | None = None  # None = config / mode preset
    use_tools: bool = True
    verify: bool = True  # run-and-repair for Python answers
    tests: str | None = None  # known tests (evals); code is saved as solution.py
    repair_attempts: int | None = None  # None = config
    auto_approve_plan: bool = False  # build mode: skip the approval question
    git: bool = False  # build mode: opt-in git checkpoints
    log: bool = True


@dataclass
class TaskOutcome:
    answer: str = ""
    draft: str = ""
    thinking: str = ""
    stats: GenStats = field(default_factory=GenStats)
    tool_events: list[ToolEvent] = field(default_factory=list)
    verify: VerifyReport | None = None
    compute: ComputeReport | None = None
    build: BuildReport | None = None
    governor: GovernResult | None = None
    interrupted: bool = False
    error: str | None = None
    notes: list[str] = field(default_factory=list)
    record: TaskRecord | None = None


class TaskPipeline:
    def __init__(
        self,
        cfg: Config,
        agent: Agent,
        workspace: Workspace,
        ui: UI,
        *,
        store: LogStore | None = None,
        governor: Governor | None = None,
    ) -> None:
        self.cfg = cfg
        self.agent = agent
        self.workspace = workspace
        self.ui = ui
        self.store = store
        self.governor = governor

    # -- entry point -------------------------------------------------------------------------

    def run(self, session: Session, prompt: str, opts: TaskOptions | None = None) -> TaskOutcome:
        opts = opts or TaskOptions()
        preset = MODES[session.mode]
        think = resolve_think(self.cfg.chat.think, opts.think, preset)
        attempts = (
            opts.repair_attempts
            if opts.repair_attempts is not None
            else self.cfg.verify.repair_attempts
        )
        if session.mode == "build":
            outcome = self._build(session, prompt, think, opts)
        elif session.mode == "code":
            outcome = self._code(session, prompt, think, opts, attempts)
        elif session.mode == "compute":
            outcome = self._compute(session, prompt, think, opts)
        else:
            outcome = self._plain(session, prompt, think, opts)
        outcome.draft = outcome.answer
        self._warn_about_turn(outcome)
        if self._should_govern(session, outcome):
            self._govern(session, prompt, think, outcome, opts)
        outcome.record = self._record(session, prompt, think, opts, outcome)
        if opts.log and self.store is not None and outcome.record is not None:
            self.store.append(outcome.record)
        return outcome

    # -- modes -------------------------------------------------------------------------------

    def _from_turn(self, outcome: TaskOutcome, turn: TurnResult) -> TaskOutcome:
        outcome.answer = turn.answer
        outcome.thinking = turn.thinking
        outcome.stats += turn.stats
        outcome.tool_events = turn.tool_events
        outcome.interrupted = turn.interrupted
        outcome.error = turn.error
        if turn.truncated:
            outcome.notes.append(
                "The answer was cut off at the output limit; raise ollama.num_predict."
            )
        if turn.step_limit_hit:
            outcome.notes.append(
                f"Stopped after {self.cfg.tools.max_steps} tool steps (tools.max_steps)."
            )
        return outcome

    def _plain(self, session: Session, prompt: str, think: bool, opts: TaskOptions) -> TaskOutcome:
        turn = self.agent.run_turn(session, prompt, think=think, use_tools=opts.use_tools)
        return self._from_turn(TaskOutcome(), turn)

    def _compute(
        self, session: Session, prompt: str, think: bool, opts: TaskOptions
    ) -> TaskOutcome:
        if not opts.use_tools:
            return self._plain(session, prompt, think, opts)
        report = run_compute_task(self.agent, session, prompt, think=think, ui=self.ui)
        outcome = self._from_turn(TaskOutcome(), report.turn)
        outcome.compute = report
        if report.evidence:
            self.ui.preview("Computation (code and result)", report.evidence)
        if report.consistent is False:
            self.ui.tag("WARN", "The final answer does not quote the last computed value.")
        return outcome

    def _code(
        self, session: Session, prompt: str, think: bool, opts: TaskOptions, attempts: int
    ) -> TaskOutcome:
        if opts.tests is not None:
            report = run_python_task(
                self.agent,
                session,
                prompt,
                tests=opts.tests,
                attempts=attempts,
                think=think,
                ui=self.ui,
            )
            outcome = TaskOutcome(answer=report.answer, verify=report)
            outcome.stats += report.stats
            outcome.interrupted, outcome.error = report.interrupted, report.error
            return outcome
        turn = self.agent.run_turn(session, prompt, think=think, use_tools=opts.use_tools)
        outcome = self._from_turn(TaskOutcome(), turn)
        if (
            opts.verify
            and not turn.interrupted
            and not turn.error
            and extract_python(turn.answer) is not None
        ):
            report = verify_and_repair(
                self.agent, session, turn, tests=None, attempts=attempts, think=think, ui=self.ui
            )
            outcome.verify = report
            outcome.answer = report.answer
            outcome.stats = report.stats  # already includes the first generation
            outcome.interrupted = report.interrupted
            outcome.error = report.error
        return outcome

    def _build(self, session: Session, prompt: str, think: bool, opts: TaskOptions) -> TaskOutcome:
        builder = Builder(self.agent, self.workspace, self.ui, git=opts.git)
        report = builder.build(
            prompt,
            think=think,
            auto_approve=opts.auto_approve_plan,
            fix_rounds=self.cfg.verify.repair_attempts,
        )
        text = report.render()
        self.ui.preview("Build summary", text)
        session.messages += [Message("user", prompt), Message("assistant", text)]
        session.turns += 1
        outcome = TaskOutcome(answer=text, build=report, interrupted=report.interrupted)
        outcome.stats += report.stats
        return outcome

    # -- after the model ---------------------------------------------------------------------

    def _warn_about_turn(self, outcome: TaskOutcome) -> None:
        if outcome.error:
            self.ui.tag("FAIL", outcome.error)
        for note in outcome.notes:
            self.ui.tag("WARN", note)

    def _should_govern(self, session: Session, outcome: TaskOutcome) -> bool:
        return (
            self.governor is not None
            and self.governor.enabled
            and session.mode != "build"
            and bool(outcome.answer.strip())
            and not outcome.interrupted
            and outcome.error is None
            and not getattr(self.ui, "stop_requested", False)
        )

    def _govern(
        self,
        session: Session,
        prompt: str,
        think: bool,
        outcome: TaskOutcome,
        opts: TaskOptions,
    ) -> None:
        assert self.governor is not None
        evidence = [e.output for e in outcome.tool_events]
        if outcome.verify and outcome.verify.attempts:
            last = outcome.verify.attempts[-1]
            evidence.append(
                f"{outcome.verify.summary_line()}\n{clip_middle(last.output.strip(), 1500)}"
            )

        def revise(revision_prompt: str) -> str | None:
            self.ui.say("[the local model is revising its answer using the critique]", dim=True)
            turn = self.agent.run_turn(session, revision_prompt, think=think, use_tools=False)
            outcome.stats += turn.stats
            _drop_exchange(session, revision_prompt)
            if turn.interrupted or turn.error or not turn.answer.strip():
                return None
            _replace_last_answer(session, turn.answer)
            return turn.answer

        result = govern(
            self.governor,
            task_type=MODES[session.mode].task_type,
            prompt=prompt,
            draft=outcome.answer,
            tool_results=evidence,
            revise=revise,
        )
        outcome.governor = result
        if not result.used or result.final == outcome.answer:
            return
        if outcome.verify is not None and outcome.verify.passed:
            code = extract_python(result.final)
            still_ok = code is not None and run_solution(code, opts.tests, self.cfg).passed
            if not still_ok:
                self.ui.tag(
                    "WARN", "The revision failed the checks the draft passed; keeping the draft."
                )
                _replace_last_answer(session, outcome.answer)
                return
        outcome.answer = result.final

    def _record(
        self, session: Session, prompt: str, think: bool, opts: TaskOptions, outcome: TaskOutcome
    ) -> TaskRecord:
        preset = MODES[session.mode]
        sampling = sampling_for(self.cfg, preset, think)
        settings = {
            "think": think,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "top_k": sampling.top_k,
            "num_ctx": sampling.num_ctx,
            "num_predict": sampling.num_predict,
            "keep_alive": self.cfg.ollama.keep_alive,
            "tools": opts.use_tools,
            "max_steps": self.cfg.tools.max_steps,
        }
        test_results = None
        verified: dict[str, object] = {"method": None, "passed": None}
        if outcome.verify and outcome.verify.attempts:
            last = outcome.verify.attempts[-1]
            test_results = {
                "passed": outcome.verify.passed,
                "method": outcome.verify.method,
                "runner": last.runner,
                "attempts": len(outcome.verify.attempts),
                "output": clip_middle(last.output, 2000),
            }
            method = (
                outcome.verify.method if outcome.verify.passed else "tests" if opts.tests else None
            )
            verified = {"method": method, "passed": outcome.verify.passed}
        elif outcome.build and outcome.build.tests is not None:
            tests = outcome.build.tests
            test_results = {
                "passed": outcome.build.passed,
                "runner": tests.runner,
                "fix_rounds": outcome.build.fix_rounds,
                "output": clip_middle(tests.output, 2000),
            }
            verified = {"method": "build_tests", "passed": outcome.build.passed}
        return TaskRecord(
            mode=session.mode,
            model=self.agent.client.model,  # the model really used (it can change mid-session)
            prompt=prompt,
            final_answer=outcome.answer,
            system=session.system_prompt(),
            settings=settings,
            local_draft=outcome.draft,
            thinking=outcome.thinking,
            tool_calls=[e.to_dict() for e in outcome.tool_events],
            test_results=test_results,
            governor=(
                outcome.governor.to_log(self.cfg.governor.claude_model)
                if outcome.governor
                else None
            ),
            verified=verified,
            stats=outcome.stats.to_dict(),
        )


def _drop_exchange(session: Session, revision_prompt: str) -> None:
    """Remove the revision request and everything after it from the history."""
    while session.messages:
        message = session.messages.pop()
        if message.role == "user" and message.content == revision_prompt:
            return


def _replace_last_answer(session: Session, text: str) -> None:
    if session.messages and session.messages[-1].role == "assistant":
        session.messages[-1] = Message("assistant", text)
    else:
        session.messages.append(Message("assistant", text))
