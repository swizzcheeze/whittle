"""
Whittle interactive installer.

Run:  python install.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MCP_DIR = ROOT / "mcp_server"
VENV = MCP_DIR / ".venv"
CONFIG_PATH = MCP_DIR / "whittle.config.json"

WIN = sys.platform == "win32"
VENV_PY = VENV / ("Scripts/python.exe" if WIN else "bin/python")
VENV_PIP = VENV / ("Scripts/pip.exe" if WIN else "bin/pip")
VENV_EXE = VENV / ("Scripts/curator-mcp.exe" if WIN else "bin/curator-mcp")


# ── tiny helpers ──────────────────────────────────────────────────────────────

def _hr():
    print("\n" + "-" * 58)


def ask(prompt: str, *, default: str | None = None, options: list[str] | None = None) -> str:
    """Read a line with an optional default and optional constrained options."""
    if options:
        rendered = "/".join(f"[{o}]" if o == default else o for o in options)
        full = f"  {prompt} ({rendered}): "
    elif default is not None:
        full = f"  {prompt} [{default}]: "
    else:
        full = f"  {prompt}: "

    while True:
        raw = input(full).strip()
        val = raw or default or ""
        if not val:
            print("    (required - please enter a value)")
            continue
        if options:
            match = next((o for o in options if o.lower() == val.lower()), None)
            if not match:
                print(f"    Please choose one of: {', '.join(options)}")
                continue
            return match
        return val


def run(cmd: list, *, hide: bool = True, check: bool = True) -> bool:
    """Run a subprocess. Returns True on success, False on failure."""
    kwargs: dict = {"check": False}
    if hide:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.PIPE
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        if check:
            stderr_text = (result.stderr or b"").decode(errors="replace").strip()
            if stderr_text:
                print(f"\n    ERROR output:\n{stderr_text}")
            raise subprocess.CalledProcessError(result.returncode, cmd)
        return False
    return True


def step(n: int, total: int, label: str) -> None:
    print(f"\n[{n}/{total}] {label}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 58)
    print("  Whittle - dataset curation workspace setup")
    print("=" * 58)

    if sys.version_info < (3, 10):
        print(f"\nERROR: Python 3.10+ required. You have {sys.version}.")
        sys.exit(1)
    print(f"\nPython {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} - OK")

    # ── 1. Mode ───────────────────────────────────────────────────────────────
    _hr()
    print("\nHow do you want to use Whittle?\n")
    print("  mcp   - wire it into an LLM client (Claude, LM Studio, etc.)")
    print("  cli   - run curate.py from the terminal yourself")
    print("  both  - install everything")
    mode = ask("Mode", default="mcp", options=["mcp", "cli", "both"])

    # ── 2. Viewer (Spotlight) ─────────────────────────────────────────────────
    want_viewer = False
    if mode in ("mcp", "both"):
        _hr()
        print("\nSpotlight visual viewer adds a browser-based scatter/table view.")
        print("It takes ~2 min to install (heavy deps: torch, av, etc.).")
        want_viewer = ask("Install Spotlight viewer?", default="yes", options=["yes", "no"]) == "yes"

    # ── 3. Embedding backend ──────────────────────────────────────────────────
    _hr()
    print("\nEmbedding backend - where should text -> vector calls go?\n")
    print("  ollama    - local Ollama daemon (default; bge-m3 recommended)")
    print("  lmstudio  - LM Studio's local server (/v1/embeddings)")
    print("  openai    - any OpenAI-compatible endpoint (cloud or self-hosted)")
    backend_choice = ask("Backend", default="ollama", options=["ollama", "lmstudio", "openai"])

    if backend_choice == "ollama":
        base_url   = ask("Ollama URL", default="http://localhost:11434")
        model      = ask("Model name", default="bge-m3")
        backend    = "ollama"
        api_key    = None
        batch_size = None   # Ollama is one-at-a-time; not written to config

    elif backend_choice == "lmstudio":
        base_url = ask("LM Studio URL", default="http://localhost:1234/v1")
        print("\n  Tip: load an embedding model in LM Studio first, then copy its")
        print("  identifier (e.g. nomic-embed-text, text-embedding-nomic-embed-text-v1.5).")
        model      = ask("Model identifier", default="nomic-embed-text")
        print("\n  Batch size: texts sent per /v1/embeddings call. 64 is safe for most")
        print("  LM Studio setups; raise to 128-256 if your machine has headroom.")
        batch_size = int(ask("Embedding batch size", default="64"))
        backend    = "openai"   # LM Studio uses the OpenAI-compatible API
        api_key    = None

    else:  # openai / custom
        base_url   = ask("Base URL", default="https://api.openai.com/v1")
        model      = ask("Model name", default="text-embedding-3-small")
        print("\n  Batch size: texts sent per /v1/embeddings call.")
        batch_size = int(ask("Embedding batch size", default="64"))
        api_key_input = ask("API key (or press Enter to read from OPENAI_API_KEY env var)", default="")
        api_key    = api_key_input or None
        backend    = "openai"

    # ── 4. Auto-load ──────────────────────────────────────────────────────────
    _hr()
    print("\nAuto-load: the server can load a dataset immediately on startup.")
    print("This prevents 'No dataset loaded' errors when the MCP client restarts")
    print("the server process mid-session (a known LM Studio quirk).\n")
    want_autoload = ask("Auto-load a dataset on startup?", default="no", options=["yes", "no"]) == "yes"
    autoload_cfg: dict | None = None
    if want_autoload:
        ds_path  = ask("Absolute path to dataset (.csv or .jsonl)")
        text_col = ask("Text column name", default="text")
        autoload_cfg = {"path": ds_path, "text_column": text_col}

    # ── 5. Pre-warm UMAP ──────────────────────────────────────────────────────
    _hr()
    print("\nThe first `project_2d` call triggers a ~25s numba JIT compilation.")
    print("Pre-warming runs that compilation in the background at server startup,")
    print("so the real call finishes in ~1s instead of timing out.\n")
    prewarm = ask("Pre-warm UMAP on startup?", default="yes", options=["yes", "no"]) == "yes"

    # ── 6. Install ────────────────────────────────────────────────────────────
    _hr()
    total_steps = 4 if mode in ("mcp", "both") else 3
    cur = 0

    cur += 1
    step(cur, total_steps, "Creating virtual environment")
    if VENV.exists():
        print("    (already exists, skipping)")
    else:
        run([sys.executable, "-m", "venv", str(VENV)])
        print("    done.")

    cur += 1
    step(cur, total_steps, "Upgrading pip")
    ok = run([str(VENV_PIP), "install", "--upgrade", "pip"], check=False)
    print("    done." if ok else "    (skipped - pip upgrade failed, continuing anyway)")

    if mode in ("cli", "both"):
        cur += 1
        step(cur, total_steps, "Installing CLI dependencies  (pandas, umap-learn, etc.)")
        run([str(VENV_PIP), "install", "-r", str(ROOT / "requirements.txt")])
        print("    done.")

    if mode in ("mcp", "both"):
        cur += 1
        label        = "viewer extra included" if want_viewer else "core only"
        # pip editable install with extras: "path/to/pkg[extra]" - extras can't be a path component
        install_spec = f"{MCP_DIR}[viewer]" if want_viewer else str(MCP_DIR)
        step(cur, total_steps, f"Installing MCP server  ({label})")
        print("    (this may take a few minutes if Spotlight is included)")
        run([str(VENV_PIP), "install", "-e", install_spec])
        print("    done.")

    # ── 7. Write config ───────────────────────────────────────────────────────
    config: dict = {
        "embedding_backend": backend,
        "embedding_base_url": base_url,
        "embedding_model": model,
        "prewarm_umap": prewarm,
    }
    if batch_size is not None:
        config["embedding_batch_size"] = batch_size
    if api_key:
        config["embedding_api_key"] = api_key
    if autoload_cfg:
        config["auto_load"] = autoload_cfg

    CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"\nConfig saved -> {CONFIG_PATH}")

    # ── 8. Print next-steps ───────────────────────────────────────────────────
    _hr()
    print("\nSetup complete!\n")

    if mode in ("cli", "both"):
        py = VENV_PY
        print("CLI usage:")
        print(f"  {py} curate.py --input data/sample.csv --text-column text\n")

    if mode in ("mcp", "both"):
        snippet = {"whittle": {"command": str(VENV_EXE)}}
        print("Add this entry to your MCP client's config file:\n")
        print(json.dumps(snippet, indent=2))
        print()
        print("Client config locations:")
        print("  LM Studio     ->  Settings > Local Server > MCP Servers (or your mcp-servers.json)")
        print("  Claude Desktop ->  %APPDATA%\\Claude\\claude_desktop_config.json")
        print("  Claude Code   ->  .claude/settings.json in your project")
        print()
        print("Then restart your MCP client to pick up the new server.")

    if backend_choice == "ollama":
        print(f"\nRemember: Ollama must be running and the model pulled before loading data:")
        print(f"  ollama serve")
        print(f"  ollama pull {model}")

    elif backend_choice == "lmstudio":
        print(f"\nRemember: LM Studio's local server must be running with an embedding model loaded.")
        print(f"  In LM Studio: Local Server tab -> load '{model}' -> Start Server")

    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled.")
        sys.exit(130)
