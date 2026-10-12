from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_neo.config import default_config
from ai_neo.evals import (
    CATEGORIES,
    PipelineRunner,
    Report,
    RunOutput,
    SuiteError,
    Task,
    TaskResult,
    check_build,
    check_exact,
    check_regex,
    check_rubric,
    compare_to_baseline,
    evaluate,
    final_answer,
    load_suite,
    render_report,
    run_suite,
)
from ai_neo.logging_store import LogStore
from ai_neo.pipeline import TaskPipeline
from ai_neo.tools import Workspace
from ai_neo.types import GenStats
from tests.fake_ollama import FakeOllama, text
from tests.helpers import ScriptedUI, make_agent

SUITE = Path(__file__).resolve().parent.parent / "evals"
CFG = default_config()

# Reference answers for every shipped task: proves each one can be passed (and a wrong answer cannot).
TEXT_ANSWERS = {
    "reasoning-syllogism": ("Final answer: yes", "Final answer: no"),
    "reasoning-bat-and-ball": ("Final answer: 5", "Final answer: 10"),
    "reasoning-height-order": ("Final answer: Bob", "Final answer: Alice"),
    "reasoning-widgets": ("Final answer: 5", "Final answer: 100"),
    "reasoning-weekday": ("Final answer: Friday", "Final answer: Thursday"),
    "computation-multiplication": ("Final answer: 83,810,205", "Final answer: 83810025"),
    "computation-factorial": ("Final answer: 355687428096000", "Final answer: 355687428096001"),
    "computation-multiples": ("Final answer: 233168", "Final answer: 233333"),
    "computation-modular-power": ("Final answer: 698635", "Final answer: 698636"),
    "computation-compound-interest": ("Final answer: $1,628.89", "Final answer: $1,500.00"),
}
CODE_ANSWERS = {
    "python-palindrome": (
        "import re\n\ndef is_palindrome(s):\n    t = re.sub(r'[^A-Za-z0-9]', '', s).lower()\n"
        "    return t == t[::-1]\n",
        "def is_palindrome(s):\n    return s == s[::-1]\n",
    ),
    "python-fizzbuzz": (
        "def fizzbuzz(n):\n    out = []\n    for i in range(1, n + 1):\n"
        "        out.append('FizzBuzz' if i % 15 == 0 else 'Fizz' if i % 3 == 0 else "
        "'Buzz' if i % 5 == 0 else str(i))\n    return out\n",
        "def fizzbuzz(n):\n    return [str(i) for i in range(1, n + 1)]\n",
    ),
    "python-merge-intervals": (
        "def merge_intervals(intervals):\n    out = []\n    for a, b in sorted(intervals):\n"
        "        if out and a <= out[-1][1]:\n            out[-1] = (out[-1][0], max(out[-1][1], b))\n"
        "        else:\n            out.append((a, b))\n    return out\n",
        "def merge_intervals(intervals):\n    return sorted(intervals)\n",
    ),
    "python-word-count": (
        "import re\nfrom collections import Counter\n\ndef word_count(text):\n"
        "    return dict(Counter(w.lower() for w in re.findall(r'[A-Za-z0-9]+', text)))\n",
        "def word_count(text):\n    return {w: text.split().count(w) for w in text.split()}\n",
    ),
}
WRITING_ANSWERS = {
    "writing-product-description": (
        "Meet the bottle built for long days. Its double-wall insulation keeps drinks icy for "
        "twenty-four hours or steaming for twelve, so your morning coffee is still hot at lunch. "
        "The leak-proof lid seals tightly, which means it can ride in a backpack, a gym bag or a "
        "car door without a single drip. Made from durable stainless steel, it resists dents and "
        "never holds on to old flavors. The wide mouth fits ice cubes and is simple to clean, and "
        "the slim shape slides into most cup holders. Fill it once and stay refreshed all day.",
        "A revolutionary bottle.",
    ),
    "writing-decline-email": (
        "Hi Sam,\n\nThank you for the invitation to meet on Tuesday. I have a deadline that day "
        "that I cannot move, so I will not be able to attend. Would Thursday work for you instead? "
        "I am free any time after ten in the morning and happy to adjust to whatever suits your "
        "schedule best. If Thursday is difficult, send me a couple of other options and I will do "
        "my best to fit one in. Thanks for understanding.\n\nBest,\nAlex",
        "Hi Sam, unfortunately no. Alex",
    ),
    "writing-three-bullets": (
        "- Catches regressions early, before users ever see them.\n"
        "- Documents how code is meant to behave.\n"
        "- Makes refactoring safer and faster.",
        "- one\n- two\n",
    ),
    "writing-two-paragraphs": (
        "Sleep gives the brain time to sort the day. While we rest, the brain moves new "
        "information into long-term memory and clears out clutter, which is why learning "
        "sticks better after a good night. Skipping sleep makes it harder to focus and slows "
        "reaction time.\n\nSleep also steadies mood. People who rest well handle stress with "
        "more patience, and they feel more positive about ordinary problems. A tired person is "
        "more likely to snap, worry or feel low. Seven to nine hours is a simple, free way to "
        "protect both thinking and feelings.",
        "Sleep is good.",
    ),
}
BUILD_ANSWERS = {
    "build-textstats": {
        "textstats.py": (
            "def count_words(text):\n    return len(text.split())\n\n\n"
            "def most_common(text, n):\n    counts = {}\n    for w in text.lower().split():\n"
            "        counts[w] = counts.get(w, 0) + 1\n"
            "    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]\n"
        ),
    },
    "build-stack": {
        "stack.py": (
            "class Stack:\n    def __init__(self):\n        self._items = []\n\n"
            "    def push(self, item):\n        self._items.append(item)\n\n"
            "    def pop(self):\n        if not self._items:\n            raise IndexError('empty')\n"
            "        return self._items.pop()\n\n"
            "    def peek(self):\n        if not self._items:\n            raise IndexError('empty')\n"
            "        return self._items[-1]\n\n"
            "    def is_empty(self):\n        return not self._items\n\n"
            "    def __len__(self):\n        return len(self._items)\n"
        ),
    },
    "build-todo-list": {
        "todo.py": (
            "import json\n\n\nclass TodoList:\n    def __init__(self):\n        self.tasks = {}\n"
            "        self.next_id = 1\n\n"
            "    def add(self, title):\n        tid = self.next_id\n        self.next_id += 1\n"
            "        self.tasks[tid] = {'title': title, 'done': False}\n        return tid\n\n"
            "    def complete(self, task_id):\n        self.tasks[task_id]['done'] = True\n\n"
            "    def pending(self):\n        return [t['title'] for t in self.tasks.values() if not t['done']]\n\n"
            "    def save(self, path):\n        with open(path, 'w') as f:\n"
            "            json.dump({'next': self.next_id, 'tasks': {str(k): v for k, v in self.tasks.items()}}, f)\n\n"
            "    @classmethod\n    def load(cls, path):\n        with open(path) as f:\n"
            "            data = json.load(f)\n        obj = cls()\n        obj.next_id = data['next']\n"
            "        obj.tasks = {int(k): v for k, v in data['tasks'].items()}\n        return obj\n"
        ),
    },
}


def by_id() -> dict[str, Task]:
    return {t.id: t for t in load_suite(SUITE)}


def test_shipped_suite_has_three_or_more_tasks_per_category() -> None:
    tasks = load_suite(SUITE)
    for category in CATEGORIES:
        assert sum(t.category == category for t in tasks) >= 3, category
    assert len({t.id for t in tasks}) == len(tasks)
    covered = set(TEXT_ANSWERS) | set(CODE_ANSWERS) | set(WRITING_ANSWERS) | set(BUILD_ANSWERS)
    assert covered == {t.id for t in tasks}  # every shipped task has a reference answer


@pytest.mark.parametrize("task_id", sorted(TEXT_ANSWERS))
def test_reasoning_and_computation_reference_answers(task_id: str) -> None:
    good, bad = TEXT_ANSWERS[task_id]
    task = by_id()[task_id]
    assert evaluate(task, RunOutput(f"Some working.\n{good}"), CFG).passed
    assert not evaluate(task, RunOutput(bad), CFG).passed


@pytest.mark.parametrize("task_id", sorted(CODE_ANSWERS))
def test_python_reference_solutions(task_id: str) -> None:
    good, bad = CODE_ANSWERS[task_id]
    task = by_id()[task_id]
    assert evaluate(task, RunOutput(f"```python\n{good}```"), CFG).passed
    assert not evaluate(task, RunOutput(f"```python\n{bad}```"), CFG).passed
    assert not evaluate(task, RunOutput("no code"), CFG).passed


@pytest.mark.parametrize("task_id", sorted(WRITING_ANSWERS))
def test_writing_reference_texts(task_id: str) -> None:
    good, bad = WRITING_ANSWERS[task_id]
    task = by_id()[task_id]
    result = evaluate(task, RunOutput(good), CFG)
    assert result.passed, result.detail
    assert not evaluate(task, RunOutput(bad), CFG).passed


@pytest.mark.parametrize("task_id", sorted(BUILD_ANSWERS))
def test_build_reference_projects(task_id: str, tmp_path: Path) -> None:
    task = by_id()[task_id]
    project = tmp_path / "proj"
    project.mkdir()
    for name, code in BUILD_ANSWERS[task_id].items():
        (project / name).write_text(code, encoding="utf-8")
    assert evaluate(task, RunOutput("", project_dir=project), CFG).passed
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / next(iter(BUILD_ANSWERS[task_id]))).write_text("x = 1\n", encoding="utf-8")
    assert not evaluate(task, RunOutput("", project_dir=broken), CFG).passed
    assert not evaluate(task, RunOutput("", project_dir=None), CFG).passed


# -- checkers ------------------------------------------------------------------------------------


def test_final_answer_extraction() -> None:
    assert final_answer("blah\nFinal answer: 42") == "42"
    assert final_answer("**Final answer:** **42**") == "42"
    assert final_answer("a\nFinal answer: 1\nmore\nFinal answer: 2") == "2"
    assert final_answer("just text\nlast line") == "last line"
    assert final_answer("") == ""


def test_exact_numeric_and_string_rules() -> None:
    spec = {"type": "exact", "answer": "1628.89", "numeric": True, "tolerance": 0.005}
    assert check_exact("Final answer: $1,628.894", spec).passed
    assert not check_exact("Final answer: 1628.90", spec).passed
    assert not check_exact("Final answer: lots", spec).passed
    word = {"type": "exact", "answer": "yes"}
    assert check_exact("Final answer: Yes, they are.", word).passed
    assert check_exact("Final answer: YES.", word).passed
    assert not check_exact("Final answer: no", word).passed
    phrase = {"type": "exact", "answers": ["New York", "NYC"]}
    assert check_exact("Final answer: nyc", phrase).passed
    assert not check_exact("Final answer: New", phrase).passed


def test_regex_rules() -> None:
    assert check_regex("The total is 42.", {"pattern": r"\b42\b"}).passed
    assert not check_regex("The total is 420.", {"pattern": r"\b42\b"}).passed
    assert check_regex("ABC", {"pattern": "abc", "flags": "i"}).passed
    assert check_regex("lots\nFinal answer: 7", {"pattern": r"^7$", "on": "final"}).passed


def test_rubric_rules_report_each_failure() -> None:
    spec = {
        "min_words": 5,
        "max_words": 8,
        "must_include": ["cat", ["dog", "hound"]],
        "banned": ["very"],
        "must_match": [r"\.\Z"],
        "paragraphs": 1,
    }
    assert check_rubric("The cat and the dog nap.", spec).passed
    result = check_rubric("A very long sentence about the cat chasing nothing at all today", spec)
    assert not result.passed
    for fragment in ("too long", "missing required point", "banned phrase", "required pattern"):
        assert fragment in result.detail
    bullets = {"bullets": {"exact": 2, "max_words_each": 3}}
    assert check_rubric("- one two\n- three", bullets).passed
    assert not check_rubric("- one two three four\n- x", bullets).passed
    assert not check_rubric("- only one", bullets).passed


def test_build_check_rejects_escaping_test_paths(tmp_path: Path) -> None:
    project = tmp_path / "p"
    project.mkdir()
    result = check_build(project, {"tests": {"../evil_test.py": "x"}}, CFG)
    assert not result.passed and "bad acceptance test path" in result.detail
    assert not (tmp_path / "evil_test.py").exists()


# -- loading -------------------------------------------------------------------------------------


def test_load_suite_reports_all_problems(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text(
        json.dumps(
            {
                "category": "reasoning",
                "tasks": [
                    {"id": "x", "prompt": "p", "check": {"type": "magic"}},
                    {"id": "y", "prompt": "p", "check": {"type": "regex", "pattern": "("}},
                    {"id": "z", "prompt": "p", "check": {"type": "python_tests", "tests": "t"}},
                    {"id": "y", "prompt": "dup", "check": {"type": "exact", "answer": "1"}},
                    {"prompt": "no id"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "b.json").write_text('{"category": "poetry", "tasks": []}', encoding="utf-8")
    (tmp_path / "c.json").write_text("{nope", encoding="utf-8")
    with pytest.raises(SuiteError) as info:
        load_suite(tmp_path)
    message = str(info.value)
    for fragment in (
        "check type must be one of",
        "bad regex",
        "does not fit category",
        "'category' must be one of",
        "cannot read as JSON",
        "needs an 'id'",
    ):
        assert fragment in message


def test_load_suite_filters_and_handles_missing(tmp_path: Path) -> None:
    assert {t.category for t in load_suite(SUITE, ["python"])} == {"python"}
    with pytest.raises(SuiteError, match="No task files"):
        load_suite(tmp_path / "nothing")


# -- running and reporting -------------------------------------------------------------------------


class StubRunner:
    def __init__(self, answers: dict[str, str], tok_s: float = 10.0) -> None:
        self.answers = answers
        self.tok_s = tok_s

    def run(self, task: Task) -> RunOutput:
        stats = GenStats(eval_count=int(self.tok_s * 2), eval_duration_ns=2_000_000_000)
        answer = self.answers.get(task.id, "")
        code = answer if task.category == "python" else None
        return RunOutput(answer, stats, code=code)


def test_run_suite_pass_rates_and_report(tmp_path: Path) -> None:
    tasks = [t for t in load_suite(SUITE) if t.category in ("reasoning", "computation")]
    answers = {tid: good for tid, (good, _) in TEXT_ANSWERS.items()}
    answers["reasoning-syllogism"] = "Final answer: no"  # one deliberate miss
    ui = ScriptedUI()
    report = run_suite(tasks, StubRunner(answers), CFG, ui=ui, suite_name="evals/")
    stats = report.category_stats()
    assert stats["reasoning"]["passed"] == 4 and stats["reasoning"]["total"] == 5
    assert stats["reasoning"]["pass_rate"] == 0.8 and stats["computation"]["pass_rate"] == 1.0
    assert stats["computation"]["mean_tok_s"] == 10.0
    assert report.overall()["passed"] == 9
    data = report.to_dict()
    assert data["suite"] == "evals/" and len(data["tasks"]) == 10
    assert "[FAIL] reasoning-syllogism" in ui.text and "[PASS]" in ui.text
    render_report(data, ui)
    assert "OVERALL" in ui.text and "80%" in ui.text


def test_baseline_ratio_and_target() -> None:
    current = {
        "overall": {"pass_rate": 0.6},
        "categories": {
            "python": {"pass_rate": 0.9},
            "writing": {"pass_rate": 0.5},
            "build": {"pass_rate": 0.3},
        },
    }
    baseline = {
        "overall": {"pass_rate": 0.8},
        "categories": {
            "python": {"pass_rate": 1.0},
            "writing": {"pass_rate": 1.0},
            "build": {"pass_rate": 0.0},
        },
    }
    out = compare_to_baseline(current, baseline, 0.9)
    assert out["categories"]["python"] == {
        "pass_rate": 0.9,
        "baseline_pass_rate": 1.0,
        "ratio": 0.9,
        "meets_target": True,
    }
    assert out["categories"]["writing"]["meets_target"] is False
    assert out["categories"]["build"]["ratio"] is None  # baseline of 0 gives no ratio
    assert out["overall"]["ratio"] == 0.75 and not out["overall"]["meets_target"]
    ui = ScriptedUI()
    report = Report("s", "m", "t", [TaskResult("a", "python", True, "tests", "", 5.0, 1.0, "")])
    render_report(report.to_dict(baseline=baseline), ui)
    assert "ratio vs baseline" in ui.text


def test_run_suite_logs_verified_exact_matches(tmp_path: Path) -> None:
    from ai_neo.logging_store import TaskRecord

    class Runner:
        def run(self, task: Task) -> RunOutput:
            record = TaskRecord(
                mode="reason", model="m", prompt=task.prompt, final_answer="Final answer: 5"
            )
            return RunOutput("Final answer: 5", record=record)

    tasks = [t for t in load_suite(SUITE) if t.id == "reasoning-bat-and-ball"]
    store = LogStore(tmp_path / "logs")
    run_suite(tasks, Runner(), CFG, ui=ScriptedUI(), store=store)
    (rec,) = list(store.records())
    assert rec["verified"] == {"method": "exact_match", "passed": True}
    assert store.export_dataset(tmp_path / "d.jsonl").exported == 1


# -- the real pipeline runner against the fake Ollama --------------------------------------------


def test_pipeline_runner_end_to_end(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    fake_ollama.queue(  # suite files load alphabetically: python before reasoning
        text("```python\n" + CODE_ANSWERS["python-fizzbuzz"][0] + "```"),
        text("The ball costs 5 cents.\nFinal answer: 5"),
    )
    agent, ui, cfg = make_agent(fake_ollama, tmp_path)
    pipeline = TaskPipeline(cfg, agent, Workspace(cfg.workspace), ui)
    tasks = [t for t in load_suite(SUITE) if t.id in ("reasoning-bat-and-ball", "python-fizzbuzz")]
    report = run_suite(tasks, PipelineRunner(pipeline, repair_attempts=0), cfg, ui=ui)
    assert [r.passed for r in report.results] == [True, True]
    assert "solution.py" in fake_ollama.chat_payloads()[0]["messages"][-1]["content"]
    asked = fake_ollama.chat_payloads()[1]["messages"][-1]["content"]
    assert asked.endswith("`Final answer: <answer>`.")  # answer format requested for exact tasks
