# Setting up AI Neo (for absolute beginners)

This guide covers everything you do **before** AI Neo can run. Once Python is installed and AI Neo
is installed, the program itself takes over: `ai-neo setup` walks you through the rest (Ollama, the
model, your settings) one step at a time.

The main part is for **Windows 11**. Linux and macOS commands are in [section 2](#2-linux-and-macos).

**What you will have at the end:** Python, a private "virtual environment" for AI Neo, the `ai-neo`
command, and a working local AI assistant. Plan on 20-40 minutes, mostly waiting for the 19 GB model
download.

---

## 1. Windows 11

### Step 1: Install Python 3.11 or newer

1. Open <https://www.python.org/downloads/windows/> in your browser and download the latest
   **Python 3.12 or 3.13** installer (64-bit). Version 3.11 or newer is required.
2. Run the installer. On the **very first screen**, tick the box
   **"Add python.exe to PATH"**, then click **Install Now**. (If you forget this box, nothing will
   work later. If you did forget, run the installer again and choose *Modify*, or see the problems
   table below.)
3. When it says *Setup was successful*, close the installer.

Check that it worked. Open **PowerShell** (press the Windows key, type `PowerShell`, press Enter) and
type:

```powershell
python --version
```

You should see something like `Python 3.12.4`. If you see an error instead, jump to
[Common problems](#3-common-problems).

### Step 2: Get the AI Neo files

If you have Git:

```powershell
git clone https://github.com/Eklovely/Eklovely.git
```

No Git? On the repository's GitHub page click **Code -> Download ZIP**, then right-click the ZIP and
choose **Extract All**. Either way you end up with a folder that contains an `ai-neo` folder.

### Step 3: Open PowerShell in the project folder

Use `cd` ("change directory") to go into the `ai-neo` folder. Adjust the path to where you put it:

```powershell
cd $HOME\Eklovely\ai-neo
```

Tip: in File Explorer you can open the `ai-neo` folder, click the address bar, type `powershell` and
press Enter to open PowerShell right there.

You should now see the folder name in the prompt, and `dir` should list `pyproject.toml`.

### Step 4: Create and activate a virtual environment

A virtual environment is a private box of Python packages, so AI Neo cannot disturb anything else on
your computer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

When it works, your prompt starts with `(.venv)`.

**If PowerShell says "running scripts is disabled on this system"**, allow scripts for your own
account only (this is a standard, safe setting) and try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

(Prefer not to change it? Use the classic Command Prompt instead: `.venv\Scripts\activate.bat`.)

You must activate the environment **every time you open a new PowerShell window** before using
`ai-neo`.

### Step 5: Install AI Neo

```powershell
python -m pip install --upgrade pip
pip install -e .
```

This installs AI Neo and its two small helpers. Want the optional Claude "governor" too (it needs an
Anthropic account and costs money; you can skip it)? Install it with the extra instead:

```powershell
pip install -e .[claude]
```

Check:

```powershell
ai-neo --version
```

### Step 6: Run the guided setup

```powershell
ai-neo setup
```

The wizard takes nine short steps. For each one it explains **what** it does and **why**, shows the
exact command before running anything, checks the result with `[OK]`, `[WARN]` or `[FAIL]`, and
if something fails it lists the likely causes and lets you **[R]etry, [S]kip or [Q]uit**. It asks your
permission before downloading anything or running a command, and it never installs software for
you; when Ollama is missing it shows you <https://ollama.com/download> and waits for you to press
Enter.

The steps: 1 system check -> 2 Ollama installed? -> 3 Ollama running? -> 4 model installed (it offers
to download `qwen3:30b`, about 19 GB) -> 5 first test -> 6 choose settings -> 7 optional Claude
governor (default **No**) -> 8 self-test -> 9 done.

You can press **Ctrl+C** or choose **Quit** at any time. Run `ai-neo setup` again and it skips the
steps you already finished. It is safe to run as often as you like.

### Step 7: Start using it

```powershell
ai-neo
```

Type a question and press Enter. `/help` lists the commands, **Ctrl+C** stops a reply that is taking
too long, and `/quit` leaves. Later you can check on everything with `ai-neo doctor`.

Daily use, in short: open PowerShell, `cd` to the `ai-neo` folder, run `.\.venv\Scripts\Activate.ps1`,
then `ai-neo`.

> **About speed:** `qwen3:30b` is large. On a laptop with a small GPU most of it runs on the CPU, so
> expect roughly 8-15 words per second, and a slow first answer while the model loads into memory.
> `/think off` makes answers much faster (the model skips its step-by-step thinking).

---

## 2. Linux and macOS

Install Python 3.11+ (macOS: <https://www.python.org/downloads/macos/> or `brew install python`;
Ubuntu/Debian: `sudo apt install python3 python3-venv python3-pip`). Then:

```bash
python3 --version                    # must be 3.11 or newer
git clone https://github.com/Eklovely/Eklovely.git
cd Eklovely/ai-neo
python3 -m venv .venv
source .venv/bin/activate            # your prompt now starts with (.venv)
python -m pip install --upgrade pip
pip install -e .                     # or: pip install -e '.[claude]'
ai-neo --version
ai-neo setup
```

Install Ollama from <https://ollama.com/download> (on Linux the install script from that page
starts a background service; on macOS, open the Ollama app). Then `ai-neo setup` does the rest, or
download the model yourself with `ollama pull qwen3:30b`.

To set the optional governor key, add this line to `~/.zshrc` (macOS) or `~/.bashrc` (Linux), then
open a **new** terminal:

```bash
export ANTHROPIC_API_KEY="your-key"
```

---

## 3. Common problems

| Problem | What you see | Fix |
|---------|--------------|-----|
| **Python not found** | `'python' is not recognized as an internal or external command`, or the Microsoft Store opens | Reinstall from python.org with **"Add python.exe to PATH"** ticked, then open a *new* PowerShell. Or try `py --version`; if that works, use `py` wherever this guide says `python`. If the Store opens, turn off the "python.exe" entries under *Settings -> Apps -> Advanced app settings -> App execution aliases*. |
| **Python is too old** | `requires a different Python: 3.9.x not in '>=3.11'` | Install Python 3.11 or newer from python.org. Delete the `.venv` folder and redo Step 4. |
| **Activation blocked** | `Activate.ps1 cannot be loaded because running scripts is disabled` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, answer `Y`, then activate again. Or use `.venv\Scripts\activate.bat` in Command Prompt. |
| **`ai-neo` is not recognized** | `ai-neo : The term 'ai-neo' is not recognized` | The virtual environment is not active. Run `.\.venv\Scripts\Activate.ps1` first (look for `(.venv)` in the prompt). Fallback: `python -m ai_neo`. |
| **`pip install` fails** | red errors mentioning a proxy, SSL or "no matching distribution" | Check your internet or VPN, run `python -m pip install --upgrade pip`, and make sure `python --version` is 3.11+. Behind a company proxy, ask IT for the pip proxy settings. |
| **Ollama not running** | `[FAIL] Ollama server: no answer at http://localhost:11434` | Start the **Ollama** app from the Start menu (llama icon in the system tray), or run `ollama serve` in a separate PowerShell and leave it open. Then `ai-neo doctor`. |
| **Ollama not installed / not found** | `Ollama was not found` | Install from <https://ollama.com/download>, then open a **new** PowerShell so PATH refreshes. The usual folder is `%LOCALAPPDATA%\Programs\Ollama`. |
| **Model not found** | `The model 'qwen3:30b' is not installed` / HTTP 404 | Run `ollama pull qwen3:30b` (about 19 GB), or let `ai-neo setup` do it. If the download stops halfway, run it again: it resumes. |
| **Out of memory, or very slow** | answers crawl, the computer freezes, or Ollama reports an out-of-memory error | Close other programs (browsers use a lot of RAM). Use `/think off`. Lower the context: run `ai-neo setup` and choose `4096`. 24 GB of RAM is the recommended minimum for this model; with less, use a smaller model (`ai-neo --model qwen3:8b`, after `ollama pull qwen3:8b`) or a remote Ollama server. The first answer after a pause is always slow because the model reloads; `keep_alive` keeps it loaded for 30 minutes. |
| **Port 11434 in use** | Ollama says the address is already in use | Usually another copy of Ollama is already running, which is fine. Otherwise find the program using the port: `netstat -ano \| findstr 11434`. To use another port, set the environment variable `OLLAMA_HOST=127.0.0.1:11435`, restart Ollama, and tell AI Neo: `ai-neo --base-url http://127.0.0.1:11435` (or `base_url` in `config.toml`). |
| **Firewall prompt** | Windows Defender Firewall asks about `ollama.exe` or `python.exe` | Allow access on **private networks**. Everything stays on your computer, because Ollama listens only on `localhost`. You can click Cancel and it still works locally. |
| **Download is slow or fails** | the progress bar stalls | Check internet/VPN and free disk space (about 25 GB). A proxy must be configured for the **Ollama server** itself (the `HTTPS_PROXY` environment variable, then restart Ollama), not for AI Neo. Choose **Retry**; it resumes. |
| **Bad `config.toml`** | `The configuration has problems:` followed by a list | Each line names the setting and what is allowed. Fix it in the file, or delete `config.toml` and run `ai-neo setup` again. The file lives in `%APPDATA%\AINeo`. |
| **Governor key not seen** | `ANTHROPIC_API_KEY is not set in this window yet` | `setx` only affects **new** windows. Close PowerShell, open a new one, activate the environment, and run `ai-neo setup` again. Never type the key into AI Neo itself. |

Still stuck? Run `ai-neo doctor`. It prints PASS / WARN / FAIL for every part of the setup with a
one-line fix for each failure.

---

## 4. Where AI Neo keeps things

| What | Windows | Linux / macOS |
|------|---------|---------------|
| settings, saved chats, logs, setup progress | `%APPDATA%\AINeo\` | `~/.ai_neo/` |
| projects the assistant builds (the *workspace*) | `C:\Users\<you>\AI-Neo-Workspace` | `~/AI-Neo-Workspace` |

To start completely fresh, delete the `AINeo` (or `.ai_neo`) folder; your workspace is separate.
