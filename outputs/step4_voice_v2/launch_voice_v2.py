"""Run eval/run_voice.py unchanged, with its output paths redirected to v2.

Only the module-level output constants are rebound; CLIPS (the Prompt B5
mapping), the agent wiring, the STT/TTS calls, and the stopping/scoring rules
are exactly the ones in eval/run_voice.py.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import eval.run_voice as runner  # noqa: E402

OUT = ROOT / "outputs" / "step4_voice_v2"
runner.OUT_DIR = OUT
runner.AUDIO_DIR = OUT / "audio"
runner.TRANSCRIPTS_PATH = OUT / "transcripts.jsonl"
runner.MANIFEST_PATH = OUT / "manifest.json"
runner.TICKETS_PATH = OUT / "tickets_created.csv"
runner.COMMAND = r".\venv\Scripts\python.exe outputs\step4_voice_v2\launch_voice_v2.py"

OUT.mkdir(parents=True, exist_ok=True)
raise SystemExit(runner.run_live())
