# AI Neo

A local-first command-line assistant for five kinds of work: **reasoning, computation, Python
programming, writing, and building small software projects**. It runs on the Ollama model
`qwen3:30b` (Qwen3-30B-A3B), fully offline. An optional **governor** lets Claude critique the local
model's answers; it is **off by default** and only turns on when you explicitly ask.

* Absolute beginner on Windows 11? Start with **[docs/SETUP.md](docs/SETUP.md)** and the guided
  `ai-neo setup` wizard, which walks you through everything step by step.
* Tuned for a laptop such as an Intel Core Ultra 7 155H with 32 GB RAM and a 4 GB-VRAM GPU: most of
  the model runs on the CPU at roughly 8-15 tokens/second, so AI Neo always streams, keeps the model
  loaded (`keep_alive`), uses an 8192-token context by default and summarizes old turns before the
  context fills.

## Install

Requires Python 3.11+ and [Ollama](https://ollama.com/download).

```bash
cd ai-neo
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -e .                   # add  .[claude]  for the optional governor
ollama pull qwen3:30b              # about 19 GB; the wizard can do this for you
ai-neo setup                       # guided setup (recommended the first time)
```

Runtime dependencies are `httpx` and `rich` only (Rich is optional: without it you get plain text).
`anthropic` is needed only for the governor (`pip install -e .[claude]`).

## Quick start

```bash
ai-neo                                   # interactive chat
ai-neo ask "Why is the sky blue?"        # one question, then exit
ai-neo ask "Write is_prime(n)" --mode code
ai-neo --mode build                      # then describe a small project; you approve the plan
ai-neo doctor                            # health check: PASS / WARN / FAIL with a fix for each
```

## Modes

Each mode has its own system prompt, sampling preset and tool set, all in
[`ai_neo/prompts.py`](ai_neo/prompts.py), which is meant to be edited.

| Mode | For | Thinking by default | Tools |
|------|-----|---------------------|-------|
| `reason` | step-by-step reasoning and analysis | on | calculator |
| `compute` | exact computation | on | calculator, run_python |
| `code` | Python programming with run-and-repair | on | run_python, read/write_file, list_dir, run_tests |
| `write` | prose with explicit constraints | off | none |
| `build` | plan and build a small project | off | file tools, run_tests, run_python |

Sampling follows Qwen3's recommendations and never uses greedy decoding: thinking mode uses
temperature 0.6 / top_p 0.95 / top_k 20; non-thinking uses 0.7 / 0.8 / 20.

## Command reference

### Sub-commands

| Command | What it does |
|---------|--------------|
| `ai-neo` / `ai-neo chat` | interactive chat (offers the setup wizard on first run) |
| `ai-neo ask "..." [--mode M] [--tests FILE]` | one question; reads stdin if no question is given. Exit code 1 if the answer failed its checks |
| `ai-neo setup [--non-interactive] [--reset]` | guided setup wizard |
| `ai-neo doctor` | non-interactive health check (Python, config, RAM, disk, GPU, Ollama, model, workspace, governor, and a quick offline self-test of the tools and safety guards); prints PASS / WARN / FAIL with a one-line fix; exits non-zero on any FAIL |
| `ai-neo eval --suite evals/ --report out.json [--baseline base.json] [--category C]` | run the evaluation harness |
| `ai-neo export-dataset [--out FILE] [--include-governed]` | export verified, local-only outputs |
| `ai-neo config` | print the settings in effect |

`ai_neo` (underscore) and `python -m ai_neo` work too.

### Flags (before or after the sub-command; flags beat environment variables and config.toml)

`--config FILE`, `--base-url URL`, `--model NAME`, `--num-ctx N`, `--mode M`, `--think/--no-think`,
`--workspace DIR`, `--yes/-y`, `--govern`, `--no-verify`, `--git`, `--plain`, `--debug`.

### Chat commands

| Command | Meaning |
|---------|---------|
| `/help` | list the commands |
| `/new` | start a fresh conversation |
| `/think on\|off\|auto` | thinking on, off, or the mode's default |
| `/mode reason\|compute\|code\|write\|build` | switch mode |
| `/model <name>` | switch to another installed model |
| `/system <text>` | custom system prompt (`/system reset`, or no text to show it) |
| `/tools on\|off` | allow or forbid tool calls |
| `/govern on\|off` | the optional Claude governor |
| `/save [name]`, `/load [name]` | save or load a conversation (JSON, in your user-data folder) |
| `/export` | write the conversation as markdown |
| `/stats` | speed, tokens, context usage |
| `/config` | the settings in effect |
| `/setup`, `/doctor` | run the wizard or the health check |
| `/quit` | leave (Ctrl+D works too) |

Start a message with `"""` to paste several lines; finish with `"""`.

**Ctrl+C** during a reply stops that generation (Ollama is told to stop), keeps the partial text and
returns to the prompt. It never exits the program; use `/quit` or Ctrl+D.

While the model thinks you see dimmed text (a short live tail with Rich), collapsed into
`[thought for 12s, ~340 tokens]` once the answer starts. After each reply you get tokens/second,
token count and elapsed time.

## Where things live

| What | Where |
|------|-------|
| config | `config.toml` in the user-data folder |
| saved chats / exports | `sessions/`, `exports/` |
| task logs | `logs/ai_neo-YYYYMMDD.jsonl` |
| datasets | `datasets/` |
| setup progress | `setup_state.json` |

The user-data folder is `%APPDATA%\AINeo` on Windows and `~/.ai_neo` elsewhere; set `AI_NEO_HOME`
to move it. Nothing is ever written to the current directory. Projects and files made by the
model go in the **workspace** (default `~/AI-Neo-Workspace`).

## Configuration

`config.toml`, overridden by `AI_NEO_*` environment variables, overridden by command-line flags.
Bad values are reported all at once with the setting name and where the value came from.

```toml
[ollama]
base_url = "http://localhost:11434"   # AI_NEO_BASE_URL
model = "qwen3:30b"                   # AI_NEO_MODEL
num_ctx = 8192                        # AI_NEO_NUM_CTX
keep_alive = "30m"                    # AI_NEO_KEEP_ALIVE
num_predict = 4096                    # cap on one reply, so a runaway reply cannot take minutes
request_timeout = 600.0

[chat]
mode = "reason"                       # AI_NEO_MODE
think = "auto"                        # auto = per-mode default, or "on" / "off"   (AI_NEO_THINK)

[workspace]
path = "~/AI-Neo-Workspace"           # AI_NEO_WORKSPACE

[tools]
max_steps = 8                         # hard cap on model calls per turn (AI_NEO_MAX_STEPS)
run_python_timeout = 20.0             # seconds; also the ceiling the model cannot raise
max_output_bytes = 20000
test_timeout = 120.0
test_runner = "auto"                  # auto = pytest if installed, else the built-in runner
memory_limit_mb = 2048                # 0 = no limit (see the safety notes)

[verify]
repair_attempts = 3                   # Python repair loop

[governor]
claude_model = ""                     # or AI_NEO_CLAUDE_MODEL; there is no built-in default
max_rounds = 2
score_threshold = 8
max_calls_per_session = 6
max_tokens_per_session = 30000
max_tokens_per_call = 1500

[context]
summarize_at = 0.75                   # summarize old turns past this fraction of num_ctx
keep_last_turns = 4
```

## How it works

### Tools and the agent loop

The model may call `calculator`, `run_python`, `read_file`, `write_file`, `list_dir` and
`run_tests` in a loop of at most `tools.max_steps` model calls per turn (the last call is always
tool-free, so there is always an answer). A malformed tool call (bad JSON, unknown tool, wrong
arguments, or one the server could not parse) is returned to the model as an error so it can
retry; nothing crashes. Before `write_file`, `run_python` or `run_tests` runs you see a preview and
are asked, unless you passed `--yes`. With no terminal (pipes) and no `--yes`, those actions are
denied.

* `calculator` evaluates exact arithmetic with a whitelist built on `ast` (no `eval`). Integers and
  decimals are exact fractions; expression length, nesting and exponent size are capped.
* `run_python` runs the code in a temporary folder with a timeout (default 20 s), an output cap
  (head and tail are kept, so tracebacks survive) and a minimal environment (no API keys).
* File tools only see the workspace. Absolute paths, `..` and symlinks that resolve outside it are
  rejected.

### Verify and repair

* **code mode**: the generated Python is run (with the model's own `assert` checks, or with tests you
  provide via `--tests`) and failures are fed back, up to `verify.repair_attempts` times. The result
  is reported honestly: "PASSED on the first try", "PASSED after 2 repair attempts", or "FAILED ...
  the code below is unverified". Code with no checks at all is flagged as *not verified*.
* **compute mode**: the model is told to compute with tools; if it answers without one it is
  nudged once. The code and result are shown next to the answer, and a warning appears if the
  answer does not quote the last computed number.
* **build mode**: the model writes a short plan and file list (JSON); you approve, revise or cancel;
  files are generated one at a time (later files only see *signatures* of earlier ones, to stay
  inside the context window); tests run; failures go back to the model for up to
  `verify.repair_attempts` rounds; the final summary is built from facts (files on disk, real test
  results) and lists what works and what does not. With `--git` (opt-in, off by default) it also runs
  `git init` and commits at checkpoints inside the project folder, showing each command.

### Context management

Tokens are estimated as **characters / 4**, a rough rule that is fine for English and code (it is not
used to count against the real tokenizer, because Ollama's `prompt_eval_count` ignores cached
tokens). When the history passes 75% of `num_ctx`, the oldest turns are summarized by the model
itself (thinking off) and replaced by a short summary; the system prompt and the last
`keep_last_turns` turns stay intact, and you are told when it happens.

### The Claude governor (optional, costs money)

Flow: local draft -> Claude returns a JSON critique `{score 0-10, issues[], suggested_corrections[]}`
against a rubric for the task type -> the local model revises -> repeat up to `max_rounds` or until
the score reaches `score_threshold`.

* Off unless you pass `--govern` or type `/govern on` **and** `ANTHROPIC_API_KEY` is set **and** a
  Claude model name is configured (`governor.claude_model` or `AI_NEO_CLAUDE_MODEL`).
* AI Neo never reads, prints or logs the key; it only checks that the variable exists, and the
  Anthropic SDK reads it itself.
* Before the first call of a session you see **exactly** what will be sent and must approve it (with
  `--yes` it is shown and auto-approved). Only the prompt, the draft and tool results are sent.
* Per-session caps on calls and tokens apply; each call reserves its worst-case size so the cap
  cannot be exceeded. Any API error falls back to the local answer.
* If a revision of a code answer fails the checks the draft passed, the draft is kept.
* Claude's critique text lives in its own field in the task log, never in the dataset.

### Logs and dataset export

Every task appends one JSON line to `logs/ai_neo-YYYYMMDD.jsonl`: id, timestamp, mode, model,
settings, prompt, local draft, tool calls and results, test results, governor critique (separate
field), final answer and token/time stats. All local.

`ai-neo export-dataset` writes only the local model's own outputs that passed an automated check
(provided or project tests, or an exact-match answer). The model's own `assert` self-checks do not
qualify. Governor critique text is never included, and tasks revised after a critique are skipped
unless you pass `--include-governed`.

> **Before using any Claude-generated text for training, check Anthropic's current terms of use.**
> AI Neo keeps Claude's text out of the dataset, but it cannot check licensing for you.

### Evaluation harness

`ai-neo eval --suite evals/ --report out.json` runs JSON task files in five categories (reasoning,
computation, python, writing, build; 3-5 sample tasks each are in [`evals/`](evals)) and prints the
pass rate per category and the mean tokens/second.

Checkers: exact numeric/string answer (`Final answer: ...`), regex, Python unit tests run against
the generated code, "project builds and acceptance tests pass", and a rubric for writing (length,
required points, banned phrases, paragraph/bullet counts). Pass `--baseline base.json` (an earlier
report, for example Claude's scores) to get a per-category ratio against it and whether you reach
`--target` (default 0.9). Build tasks run in a temporary folder.

Task file format:

```json
{"category": "reasoning",
 "tasks": [{"id": "my-task", "prompt": "...",
            "check": {"type": "exact", "answer": "42", "numeric": true, "tolerance": 0}}]}
```

## Safety notes (please read)

* **`run_python` is best-effort isolation, not a security sandbox.** The code runs as you, can read
  and write any file you can, and can use the network. What AI Neo does: a temporary working
  folder, a wall-clock timeout that kills the whole process tree, an output cap, closed stdin, a
  minimal environment without your API keys, and (on Linux/macOS) CPU, memory and file-size limits.
  On Windows the memory cap uses a Job Object; that code path could not be exercised in the build
  environment (Linux), so treat it as unverified. Run only code and projects you are comfortable
  running. For real isolation use a VM or container.
* File tools are restricted to the workspace and every path is resolved and verified, but a symlink
  swapped in *between* the check and the use could still win a race. Do not point the workspace at
  a folder that untrusted programs can modify.
* `write_file`, `run_python` and `run_tests` ask first. `--yes` removes the questions: use it only
  where you trust the model's output.
* The governor sends text to Anthropic and costs money. It is off by default.
* No telemetry. The only network calls are to your configured Ollama URL and, if you enabled the
  governor, the Anthropic API.

## Using a remote Ollama (for example a rented GPU)

Run Ollama on the other machine and point `base_url` at it:

```bash
ai-neo --base-url http://my-gpu-box:11434
# or in config.toml:  [ollama] base_url = "http://my-gpu-box:11434"
```

On the server, Ollama must listen on a reachable address (`OLLAMA_HOST=0.0.0.0:11434`). Ollama has
no authentication: do not expose it to the open internet. Use a private network, a VPN, or an SSH
tunnel (`ssh -L 11434:localhost:11434 user@gpu-box`, then keep the default `base_url`). With a
remote `base_url`, the wizard and doctor skip the local RAM/disk/GPU/install checks. You can raise
`num_ctx` freely on a big GPU.

## Development

```bash
pip install -e .[dev]
ruff check . && ruff format --check .
pytest                      # no real Ollama or internet needed
```

The tests use a scripted fake Ollama HTTP server on 127.0.0.1 (streaming, `thinking`, `<think>`
tags, HTTP 400 on `think`, 404 model missing, malformed tool calls, model pulls), plus a
pseudo-terminal test that sends a real Ctrl+C (POSIX only).

## Known limitations

* Tested here against a **fake** Ollama; it was not run against a real `qwen3:30b`. Real-model
  behaviour (quality, tool-call formats, speed) may differ.
* Governor and build mode: the governor does not critique build-mode runs (no revision step there).
* The `<think>` splitter handles `<think>...</think>` in the content stream; a model that omits the
  opening tag is not detected.
* Windows-specific code (Job Object memory cap, RAM detection via `GlobalMemoryStatusEx`) was
  written from the documentation and not run on Windows.
* Token counts use a documented chars/4 estimate; very non-English text uses more tokens per
  character.
