"""Real Codex startup E2E: temporary config, local provider, no model requests."""
import fcntl
import json
import os
from pathlib import Path
import pty
import re
import select
import struct
import subprocess
import sys
import tempfile
import termios
import time

WARNINGS = {
    "Running without the shared background server": "Running without the shared background server",
    "`--dangerously-bypass-hook-trust` is enabled": "hook-trust",
    "clamping SessionEnd hook timeout": "clamping",
}
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07]*?(?:\x07|\x1b\\)")


def plain(output):
    return ANSI.sub("", output.decode(errors="replace"))


def compact(text):
    return re.sub(r"\s+", "", text)


def run_case(case, artifact_dir, root, session_id):
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 160, 0, 0))
    env = dict(os.environ, **dict(case["env"]))
    env.update(CODEX_HOME=root, TERM="xterm-256color", VLX_EXE="/usr/bin/true")
    for key in ("OPENAI_API_KEY", "CODEX_API_KEY", "CODEX_AUTH_JSON"):
        env.pop(key, None)
    launch = case["launch"]
    if case["name"] in ("resume", "fork"):
        assert session_id, "a real saved session must exist before resume/fork"
        launch = launch.replace("019e8662-62f8-7a01-a144-e26a07a5e5bb", session_id)
    process = subprocess.Popen(
        ["/bin/bash", "-c", launch], cwd=root, env=env,
        stdin=slave, stdout=slave, stderr=slave,
    )
    os.close(slave)
    output = bytearray()
    reviewed = False

    def read_for(seconds):
        nonlocal reviewed
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if not select.select([master], [], [], 0.1)[0]:
                if process.poll() is not None:
                    break
                continue
            try:
                chunk = os.read(master, 65536)
            except OSError:
                break
            if not chunk:
                break
            output.extend(chunk)
            if b"\x1b[6n" in chunk:
                os.write(master, b"\x1b[1;1R")
            if b"\x1b[c" in chunk:
                os.write(master, b"\x1b[?1;2c")
            if not reviewed and "Hooksneedreview" in compact(plain(output)):
                # These six fixture hooks execute /usr/bin/true in this temporary home only.
                reviewed = True
                os.write(master, b"\x1b[B\r")

    forced_exit = False
    try:
        read_for(7)
        if process.poll() is None:
            os.write(master, b"\x1bOQ")  # F2 reveals startup warning text in current Codex.
            read_for(1)
            if case["name"] == "baseline":
                for _ in range(2):
                    os.write(master, b"\x1b[C")
                    read_for(0.4)
            os.write(master, b"\x1b")
            read_for(0.3)
            os.write(master, b"\x03")
            read_for(5)
        if process.poll() is None:
            forced_exit = True
            process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            forced_exit = True
            process.kill()
            process.wait()
        read_for(0.1)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        os.close(master)

    decoded = plain(output)
    (artifact_dir / f"{case['name']}.log").write_text(output.decode(errors="replace"))
    (artifact_dir / f"{case['name']}.txt").write_text(decoded)
    # Ratatui writes only changed cells while switching warning pages. The unique warning markers
    # stay intact even when unchanged letters in the rest of the message are not emitted again.
    found = {warning: compact(marker) in compact(decoded) for warning, marker in WARNINGS.items()}
    ids = re.findall(r"codex resume ([0-9a-f-]{36})", decoded)
    result = dict(name=case["name"], exit_code=process.returncode, forced_exit=forced_exit,
                  reviewed_fixture_hooks=reviewed, warnings=found, output_bytes=len(output))
    print(json.dumps(result), flush=True)
    assert "OpenAI Codex" in decoded, "Codex must initialize its interface"
    assert "Welcome" not in decoded, "onboarding must not mask runtime startup warnings"
    assert "unexpected argument" not in decoded and "No saved session" not in decoded
    assert process.returncode == 0 and not forced_exit, "Codex must exit normally after Ctrl+C"
    return result, ids[-1] if ids else session_id


request = json.load(sys.stdin)
artifacts = Path(request["artifacts"])
artifacts.mkdir(parents=True, exist_ok=True)
results = []
session_id = None
with tempfile.TemporaryDirectory(prefix="vela-codex-startup-") as temporary:
    root = str(Path(temporary).resolve())
    # No credentials, no real model endpoint, and no prompts: runtime can initialize without quota.
    (Path(root) / "config.toml").write_text(
        'model_provider = "startup_fixture"\nmodel = "startup-fixture"\n'
        '[model_providers.startup_fixture]\nname = "Startup fixture"\n'
        'base_url = "http://127.0.0.1:9/v1"\nwire_api = "responses"\n'
        'requires_openai_auth = false\n'
        f'[projects.{json.dumps(root)}]\ntrust_level = "trusted"\n'
    )
    for case in request["cases"]:
        result, session_id = run_case(case, artifacts, root, session_id)
        results.append(result)
        (artifacts / "results.json").write_text(json.dumps(results, indent=2) + "\n")
assert all(results[0]["warnings"].values()), "baseline must reproduce all three warnings"
assert results[1]["reviewed_fixture_hooks"], "new hook definitions must require user review"
for result in results[1:]:
    assert not any(result["warnings"].values()), f"{result['name']}: unexpected warning"
print(f"startup E2E passed; artifacts: {artifacts}")
