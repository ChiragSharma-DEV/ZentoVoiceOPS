"""CLI entrypoint for VoiceOps 2026."""

import sys
from pathlib import Path

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baseline import run_baseline


def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py [step1 | voice <folder>]")
        sys.exit(1)

    command = sys.argv[1].lower()
    if command == "step1":
        run_baseline()
    elif command == "voice":
        from eval.run_voice import run_batch
        folder = sys.argv[2] if len(sys.argv) > 2 else "voice_clips"
        run_batch(folder=folder, agent_name="stub", language_code="en-IN")
    else:
        print(f"Unknown command: {command}")
        print("Available commands: step1, voice")
        sys.exit(1)


if __name__ == "__main__":
    main()
