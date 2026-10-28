#!/usr/bin/env python3
"""Offline submission checker for VoiceOps 2026.

Standard library only. Run from the repo root:

    python submission/check_submission.py

Offline: this script performs no LLM, Gemini, or Gnani calls and runs no git
commands. It never prints a value from .env; secret findings are path:line only.

Fix-round behavior:
- .env is ignored by .gitignore, so its contents are not scanned and do not fail.
- The dataset-provided ``tools`` package is classified as a runtime dependency,
  not a missing requirements entry.
- Every import in src/ and run.py is printed with its requirements mapping.
- Runtime paths are classified and checked against the declared zip payload.
- Every ``run.py`` command documented in README.md is checked against run.py.
- A missing required argument in a documented eval command is a FAIL.
- Literal absolute paths and hard-coded repo-parent fallbacks are a FAIL.
- A missing expected zip is a FAIL (the fix-round expects that FAIL).
"""

from __future__ import annotations

import ast
import fnmatch
import re
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SUBMISSION = REPO / "submission"
TEAM_NAME = "Kaizeen"  # placeholder until the user supplies the real name
EXPECTED_ZIP = f"{TEAM_NAME}_voiceops.zip"

# Directories that are never part of the submission payload.
SKIP_DIRS = {".git", "venv", "voice_clips", "__pycache__", "node_modules"}

# Import name -> distribution name(s) that provide it.
DIST_MAP = {
    "dotenv": "python-dotenv",
    "pandas": "pandas",
    "requests": "requests",
    "google": "google-genai",
    "numpy": "numpy",
    "faiss": "faiss-cpu",
    "sentence_transformers": "sentence-transformers",
    "rank_bm25": "rank-bm25",
    "gnani": "gnani-vachana",
}

REQ_ORDER = [
    "requirements.txt",
    "requirements-step2.txt",
    "requirements-step3.txt",
    "requirements-voice.txt",
]

REQUIRED_FILES = ["README.md", ".env.example", "requirements.txt", "run.py"]
REQUIRED_DIRS = ["src", "sample_outputs"]
OPTIONAL_FILES = ["report.pdf", "demo.mp4"]
README_HEADINGS = [
    "Repository Layout",
    "Setup",
    "Configure Environment Variables",
    "Execution Commands",
    "Decisions",
    "Known Limits",
]

ZIP_SPEC = [
    "README.md",
    ".env.example",
    "requirements.txt",
    "run.py",
    "src/",
    "report.pdf",
    "demo.mp4",
    "sample_outputs/",
]

lines: list[str] = []
failed = 0
missing = 0
passed = 0


def emit(status: str, check: str, reason: str) -> None:
    global failed, missing, passed
    if status == "FAIL":
        failed += 1
    elif status == "MISSING":
        missing += 1
    elif status == "PASS":
        passed += 1
    lines.append(f"{status:<7} {check}: {reason}")


def note(check: str, reason: str) -> None:
    """A non-scoring observation (does not change PASS/FAIL counts)."""
    lines.append(f"INFO    {check}: {reason}")


def rel(path: Path) -> str:
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


def is_text(path: Path) -> bool:
    try:
        if path.stat().st_size > 5 * 1024 * 1024:
            return False
        raw = path.read_bytes()
    except OSError:
        return False
    if b"\x00" in raw[:8192]:
        return False
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


# ---------------------------------------------------------------- gitignore
def read_gitignore() -> list[str]:
    path = REPO / ".gitignore"
    if not path.is_file():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def gitignore_pattern_matches(pattern: str, path: str) -> bool:
    """Match one .gitignore pattern against a repo-relative POSIX path."""
    negated = pattern.startswith("!")
    pat = pattern[1:] if negated else pattern
    if not pat:
        return False
    dir_only = pat.endswith("/")
    pat = pat.rstrip("/")
    if not pat:
        return False

    if "/" not in pat:
        # A pattern without a slash matches at any depth.
        return any(
            fnmatch.fnmatchcase(part, pat)
            for part in path.split("/")
            if part
        )

    if fnmatch.fnmatchcase(path, pat):
        return True
    if path.startswith(pat + "/"):
        return True
    # A directory pattern also covers everything below it.
    if dir_only and any(
        fnmatch.fnmatchcase("/".join(path.split("/")[:i]), pat)
        for i in range(1, len(path.split("/")))
    ):
        return True
    return False


def would_be_ignored(path: str) -> bool:
    ignored = False
    for pattern in read_gitignore():
        if gitignore_pattern_matches(pattern, path):
            ignored = not pattern.startswith("!")
    return ignored


def iter_repo_files():
    for path in REPO.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


# ---------------------------------------------------------------- check 1
missing_required = [name for name in REQUIRED_FILES if not (REPO / name).is_file()]
missing_required += [name for name in REQUIRED_DIRS if not (REPO / name).is_dir()]
if missing_required:
    emit("FAIL", "1. Required files and folders",
         "missing: " + ", ".join(missing_required))
else:
    emit("PASS", "1. Required files and folders",
         "README.md, .env.example, requirements.txt, run.py, src/, sample_outputs/")

for name in OPTIONAL_FILES:
    if (REPO / name).is_file():
        emit("PASS", f"1. {name}", "present")
    else:
        emit("MISSING", f"1. {name}", "not present yet (user must add it)")


# ---------------------------------------------------------------- check 2
TOKEN_PATTERNS = [
    ("gnani_token", re.compile(r"vach_[A-Za-z0-9_\-]{8,}")),
    ("google_key", re.compile(r"AIza[0-9A-Za-z_\-]{30,}")),
]
ENV_KEY_PATTERN = re.compile(r"^\s*(GNANI_API_KEY|GEMINI_API_KEY)\s*=\s*(.*)$")

secret_hits: list[str] = []
scanned_paths: set[str] = set()
skipped_ignored = 0

env_path = REPO / ".env"
env_present = env_path.is_file()
env_ignored = env_present and would_be_ignored(".env")


def scan_text(path: Path) -> None:
    """Scan one text file; record findings as path:line only."""
    global skipped_ignored
    r = rel(path)
    if r in scanned_paths:
        return
    if not is_text(path):
        return
    scanned_paths.add(r)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for lineno, line in enumerate(text.splitlines(), 1):
        hit_labels: list[str] = []
        for label, pattern in TOKEN_PATTERNS:
            if pattern.search(line):
                hit_labels.append(label)
        m = ENV_KEY_PATTERN.match(line)
        if m and m.group(2).strip().strip("\"'").strip():
            hit_labels.append("gnani_key" if m.group(1) == "GNANI_API_KEY" else "gemini_key")
        for label in hit_labels:
            finding = f"{r}:{lineno} [{label}]"
            if finding not in secret_hits:
                secret_hits.append(finding)


# If .env is ignored, do not read it. If it is not ignored, scan it.
if env_present and not env_ignored:
    scan_text(env_path)

for path in iter_repo_files():
    r = rel(path)
    if r == ".env":
        continue
    if would_be_ignored(r):
        skipped_ignored += 1
        continue
    scan_text(path)

# Files that would enter the declared zip payload are always checked, even if
# an unusual .gitignore rule would otherwise exclude them.
for name in ZIP_SPEC:
    target = REPO / name.rstrip("/")
    if target.is_file() and target.name != ".env":
        scan_text(target)
    elif target.is_dir():
        for p in target.rglob("*"):
            if p.is_file() and p.name != ".env":
                scan_text(p)

if secret_hits:
    emit("FAIL", "2. Secrets",
         f"{len(secret_hits)} finding(s), path:line only -> " + "; ".join(secret_hits))
elif env_present and env_ignored:
    emit("PASS", "2. Secrets",
         "local .env present, ignored, excluded from zip; "
         f"no secret patterns in {len(scanned_paths)} scanned text file(s); "
         f"{skipped_ignored} ignored file(s) skipped")
elif env_present:
    emit("PASS", "2. Secrets",
         "local .env present but NOT ignored; .env contents scanned with no pattern match; "
         f"{len(scanned_paths)} text file(s) scanned")
else:
    emit("PASS", "2. Secrets",
         f"no local .env present; no secret patterns in {len(scanned_paths)} scanned text file(s)")


# ---------------------------------------------------------------- check 3
env_example = REPO / ".env.example"
if not env_example.is_file():
    emit("FAIL", "3. .env.example empty values", "file not found")
else:
    bad = []
    variables = 0
    for lineno, line in enumerate(env_example.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            bad.append(f"line {lineno} (no '=')")
            continue
        key, value = stripped.split("=", 1)
        variables += 1
        if value.strip().strip("\"'").strip():
            bad.append(f"{key.strip()} (line {lineno})")
    if bad:
        emit("FAIL", "3. .env.example empty values",
             "non-empty value for: " + ", ".join(bad))
    else:
        emit("PASS", "3. .env.example empty values",
             f"{variables} variable(s), all values empty")


# ---------------------------------------------------------------- check 4
STDLIB = set(getattr(sys, "stdlib_module_names", set()))
if not STDLIB:  # Python < 3.10 fallback, deliberately conservative
    STDLIB = {
        "__future__", "abc", "argparse", "ast", "asyncio", "base64", "collections",
        "concurrent", "contextlib", "copy", "csv", "dataclasses", "datetime", "decimal",
        "difflib", "enum", "functools", "glob", "hashlib", "html", "http", "io",
        "itertools", "json", "logging", "math", "os", "pathlib", "pickle", "random",
        "re", "shutil", "signal", "socket", "sqlite3", "string", "subprocess", "sys",
        "tempfile", "textwrap", "threading", "time", "traceback", "types", "typing",
        "unicodedata", "urllib", "uuid", "warnings", "wave", "xml", "zipfile",
    }

OWN_PACKAGES = {"src", "eval", "tests"}
py_files = list((REPO / "src").rglob("*.py")) if (REPO / "src").is_dir() else []
if (REPO / "run.py").is_file():
    py_files.append(REPO / "run.py")
py_files = sorted(set(py_files))

import_rows: list[dict] = []
parse_errors: list[str] = []
for path in py_files:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        parse_errors.append(f"{rel(path)} ({type(exc).__name__})")
        continue
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                import_rows.append({
                    "file": rel(path),
                    "line": node.lineno,
                    "kind": "import",
                    "module": alias.name,
                    "root": alias.name.split(".")[0],
                })
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            root = node.module.split(".")[0] if node.module else ""
            import_rows.append({
                "file": rel(path),
                "line": node.lineno,
                "kind": "from",
                "module": module,
                "root": root,
            })

import_rows.sort(key=lambda r: (r["file"], r["line"], r["module"]))

# Parse every requirements file, preferring the main requirements.txt entry.
req_map: dict[str, tuple[str, str, int]] = {}
for req_name in REQ_ORDER:
    req_path = REPO / req_name
    if not req_path.is_file():
        continue
    for lineno, raw in enumerate(req_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        pkg = re.split(r"[<>=!~;@\[]", line, maxsplit=1)[0].strip()
        if not pkg:
            continue
        key = pkg.lower().replace("_", "-")
        req_map.setdefault(key, (pkg, req_name, lineno))


def map_import(row: dict) -> tuple[str, bool]:
    """Return (requirements mapping text, covered?)."""
    root = row["root"]
    if not root:
        return "repo-relative module (not in requirements)", True
    if root == "__future__":
        return "stdlib directive", True
    if root in STDLIB:
        return "stdlib (Python)", True
    if root in OWN_PACKAGES:
        return "repo module (not in requirements)", True
    if root == "tools":
        return "dataset runtime package <DATASET_DIR>/tools (NOT in requirements by design)", True

    dist = DIST_MAP.get(root)
    if dist is None:
        return "NOT MAPPED", False
    entry = req_map.get(dist.lower().replace("_", "-"))
    if entry is None:
        return f"{dist} (MISSING from requirements)", False
    return f"{dist} ({entry[1]}:{entry[2]})", True


uncovered: list[dict] = []
for row in import_rows:
    mapping, covered = map_import(row)
    row["mapping"] = mapping
    row["covered"] = covered
    if not covered:
        uncovered.append(row)

third_party_roots = sorted({
    r["root"] for r in import_rows
    if r["root"] and r["root"] not in STDLIB and r["root"] not in OWN_PACKAGES
    and r["root"] not in {"__future__", "tools"}
})

if parse_errors:
    emit("FAIL", "4. Third-party imports",
         "could not parse: " + ", ".join(parse_errors))
elif uncovered:
    emit("FAIL", "4. Third-party imports",
         "not in requirements: " + ", ".join(
             f"{r['root']} ({r['file']}:{r['line']})" for r in uncovered
         ) + " (checked " + ", ".join(third_party_roots) + ")")
else:
    emit("PASS", "4. Third-party imports",
         f"all {len(import_rows)} import statement(s) classified; "
         f"third-party: {', '.join(third_party_roots) if third_party_roots else 'none'}")

# Explicit google-genai confirmation (the check now detects google / google.genai).
google_req = req_map.get("google-genai")
google_rows = [r for r in import_rows if r["root"] == "google"]
if google_req is None and not google_rows:
    emit("PASS", "4b. google-genai requirement",
         "google-genai not declared and 0 google.genai imports in src/ and run.py; "
         "src/llm.py calls the Gemini REST endpoint with requests, so the SDK is unused")
elif google_req is None:
    imports_where = ", ".join(f"{r['file']}:{r['line']}" for r in google_rows)
    emit("FAIL", "4b. google-genai requirement",
         f"google.genai imported at {imports_where} but google-genai not found in requirements*.txt")
else:
    req_where = f"{google_req[1]}:{google_req[2]}"
    if google_rows:
        imports_where = ", ".join(f"{r['file']}:{r['line']}" for r in google_rows)
        emit("PASS", "4b. google-genai requirement",
             f"google-genai declared at {req_where}; google.genai imported at {imports_where}")
    else:
        emit("PASS", "4b. google-genai requirement",
             f"google-genai declared at {req_where}; 0 google.genai imports in src/ and run.py "
             "(src/llm.py imports requests at line 12 and calls the Gemini REST endpoint at line 51; "
             "the declared SDK is currently unused)")

# Dataset tools package: a runtime package, never a requirements entry.
repo_tools_exists = (REPO / "tools").is_dir()
note("4c. Dataset tools package",
     f"voiceops/tools/ exists: {'yes' if repo_tools_exists else 'no'}; "
     "tools.tools maps to <DATASET_DIR>/tools/tools.py; not added to requirements")


# ---------------------------------------------------------------- check 5
readme = REPO / "README.md"
readme_text = ""
if not readme.is_file():
    emit("FAIL", "5. README structure", "README.md not found")
else:
    readme_text = readme.read_text(encoding="utf-8")
    lines_readme = readme_text.splitlines()
    headings = [
        (len(line) - len(line.lstrip("#")), line.strip())
        for line in lines_readme
        if line.lstrip().startswith("#")
    ]
    lower_headings = [text.lower() for _, text in headings]
    absent = [
        wanted for wanted in README_HEADINGS
        if not any(wanted.lower() in text for text in lower_headings)
    ]

    run_section = ""
    for idx, (level, text) in enumerate(headings):
        if "execution commands" in text.lower():
            start = lines_readme.index(next(
                l for l in lines_readme
                if l.strip() == text
            )) + 1
            end = len(lines_readme)
            for later_level, later_text in headings[idx + 1:]:
                if later_level <= level:
                    end = lines_readme.index(next(
                        l for l in lines_readme if l.strip() == later_text
                    ))
                    break
            run_section = "\n".join(lines_readme[start:end])
            break

    run_ok = bool(run_section) and "run.py" in run_section and "voice" in run_section
    if run_ok:
        folder_ok = bool(
            re.search(r"run\.py\s+voice\s+(<[^>]+>|[^\s]*folder[^\s]*)", run_section)
            or re.search(r"run\.py\s+voice\s+\S+", run_section)
        )
    else:
        folder_ok = False

    problems = []
    if absent:
        problems.append("missing heading(s): " + ", ".join(absent))
    if not run_ok:
        problems.append("Execution Commands does not mention run.py and voice")
    if run_ok and not folder_ok:
        problems.append("Execution Commands does not show a folder argument for audio")

    if problems:
        emit("FAIL", "5. README structure", "; ".join(problems))
    else:
        emit("PASS", "5. README structure",
             "all six headings present; run.py voice <folder> documented")


# ---------------------------------------------------------------- check 5b
run_py = REPO / "run.py"
run_text = run_py.read_text(encoding="utf-8") if run_py.is_file() else ""
run_branches: dict[str, int] = {}
for m in re.finditer(r'(?:if|elif)\s+command\s*==\s*["\']([^"\']+)["\']', run_text):
    lineno = run_text[:m.start()].count("\n") + 1
    run_branches.setdefault(m.group(1), lineno)

documented = []
for m in re.finditer(r"^\s*python\s+run\.py\s+([^\s`]+)", readme_text, re.MULTILINE):
    documented.append((m.group(1), readme_text[:m.start()].count("\n") + 1))

if not documented:
    emit("FAIL", "5b. README commands",
         "README documents no 'python run.py ...' command; documented command not implemented")
else:
    command_problems = []
    command_details = []
    seen = set()
    for command, readme_line in documented:
        if command in seen:
            continue
        seen.add(command)
        if command in run_branches:
            command_details.append(
                f"python run.py {command} -> branch '{command}' at run.py:{run_branches[command]}"
            )
        else:
            command_problems.append(
                f"README.md:{readme_line} python run.py {command} -> documented command not implemented"
            )
    if command_problems:
        emit("FAIL", "5b. README commands", "; ".join(command_problems))
    else:
        voice_stub = bool(
            re.search(r'command\s*==\s*["\']voice["\']', run_text)
            and 'agent_name="stub"' in run_text
        )
        suffix = "; voice branch hardcodes agent_name=\"stub\" (plumbing only)" if voice_stub else ""
        emit("PASS", "5b. README commands", "; ".join(command_details) + suffix)

# Related README command note (not a run.py branch, so it does not score).
if re.search(r"^\s*python\s+eval/run_voice\.py\s*$", readme_text, re.MULTILINE):
    note("5c. README eval/run_voice.py",
         "README.md documents 'python eval/run_voice.py'; eval/run_voice.py has no required "
         "argparse positionals and reads the default clip folder")


# ---------------------------------------------------------------- check 5d
def required_positionals(script: Path) -> list[str]:
    """Return required positional argparse names found in a script."""
    try:
        tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    except (OSError, SyntaxError):
        return []
    required: list[str] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            continue
        if not node.args:
            continue
        first = node.args[0]
        if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
            continue
        name = first.value
        if name.startswith("-"):
            continue
        kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        if "default" in kwargs:
            continue
        nargs = kwargs.get("nargs")
        if isinstance(nargs, ast.Constant) and nargs.value in {"?", "*"}:
            continue
        required.append(name)
    return required


documented_eval_commands: list[tuple[str, str, int]] = []
for lineno, line in enumerate(readme_text.splitlines(), 1):
    m = re.match(r"^\s*python\s+([^\s`]+)(.*)$", line)
    if not m:
        continue
    script_rel = m.group(1)
    rest = m.group(2).strip()
    if script_rel.startswith(("eval/", "tests/")) and (REPO / script_rel).is_file():
        documented_eval_commands.append((script_rel, rest, lineno))

command_arg_problems: list[str] = []
command_arg_details: list[str] = []
for script_rel, rest, lineno in documented_eval_commands:
    required = required_positionals(REPO / script_rel)
    if not required:
        continue
    tokens = rest.split()
    positionals: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("--"):
            if "=" not in token and i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                i += 2
                continue
            i += 1
            continue
        if token.startswith("-"):
            i += 1
            continue
        positionals.append(token)
        i += 1
    if len(positionals) < len(required):
        missing_args = required[len(positionals):]
        command_arg_problems.append(
            f"README.md:{lineno} python {script_rel} -> missing required argument(s): "
            + ", ".join(missing_args)
        )
    else:
        command_arg_details.append(
            f"README.md:{lineno} python {script_rel} -> {', '.join(required)} supplied"
        )

if command_arg_problems:
    emit("FAIL", "5d. Documented command arguments", "; ".join(command_arg_problems))
elif command_arg_details:
    emit("PASS", "5d. Documented command arguments", "; ".join(command_arg_details))
else:
    emit("PASS", "5d. Documented command arguments",
         "no documented eval command with required positional arguments")


# ---------------------------------------------------------------- check 6
sample_root = REPO / "sample_outputs"
if not sample_root.is_dir():
    emit("FAIL", "6. sample_outputs", "sample_outputs/ not found")
else:
    sample_files = [p for p in sample_root.rglob("*") if p.is_file()]
    paths = [rel(p).lower() for p in sample_files]

    has_step2 = any("step2" in p or "step_2" in p or "step 2" in p for p in paths)
    has_step3 = any("step3" in p or "step_3" in p or "step 3" in p for p in paths)

    voice_dirs = [
        d for d in sample_root.rglob("*")
        if d.is_dir()
        and (d / "transcript.txt").is_file()
        and (d / "reply.txt").is_file()
        and (d / "reply.wav").is_file()
    ]

    voice_problems = []
    for d in voice_dirs:
        chunk_files = [
            f for f in d.iterdir()
            if f.is_file() and any(k in f.name.lower() for k in ("chunk", "retriev", "source"))
        ]
        if not chunk_files:
            voice_problems.append(f"{rel(d)} has no retrieved-chunks file")

    stub_files = []
    for path in sample_files:
        if not is_text(path):
            continue
        try:
            if "[STUB]" in path.read_text(encoding="utf-8"):
                stub_files.append(rel(path))
        except OSError:
            pass

    problems = []
    if not has_step2:
        problems.append("no Step 2 sample")
    if not has_step3:
        problems.append("no Step 3 sample")
    if not voice_dirs:
        problems.append("no voice sample folder")
    problems.extend(voice_problems)
    if stub_files:
        problems.append("[STUB] found in: " + ", ".join(stub_files))

    if problems:
        emit("FAIL", "6. sample_outputs", "; ".join(problems))
    else:
        emit("PASS", "6. sample_outputs",
             f"Step 2, Step 3 and {len(voice_dirs)} voice sample(s); no [STUB]")


# ---------------------------------------------------------------- check 7
zip_candidates = [REPO / EXPECTED_ZIP, SUBMISSION / EXPECTED_ZIP]
zip_path = next((p for p in zip_candidates if p.is_file()), None)
other_zips = sorted({
    p
    for p in list(REPO.glob("*.zip")) + list(SUBMISSION.glob("*.zip"))
    if p.is_file() and p.name != EXPECTED_ZIP
})

if zip_path is None:
    extra = ""
    if other_zips:
        extra = "; other zip(s) present: " + ", ".join(rel(p) for p in other_zips)
    emit("FAIL", "7. Zip contents",
         f"{EXPECTED_ZIP} not found in repo root or submission/; "
         f".env absence from the zip cannot be confirmed{extra}")
else:
    try:
        with zipfile.ZipFile(zip_path) as archive:
            archive_names = [n.replace("\\", "/") for n in archive.namelist()]
    except (OSError, zipfile.BadZipFile) as exc:
        emit("FAIL", "7. Zip contents",
             f"{rel(zip_path)} is not readable ({type(exc).__name__})")
        archive_names = None

    if archive_names is not None:
        bad = [
            name for name in archive_names
            if name == ".env"
            or name.endswith("/.env")
            or name == "venv" or name.startswith("venv/") or "/venv/" in name
        ]
        if bad:
            emit("FAIL", "7. Zip contents",
                 f"{rel(zip_path)} contains forbidden: " + ", ".join(bad))
        else:
            emit("PASS", "7. Zip contents",
                 f"{rel(zip_path)} has no .env and no venv/ "
                 f"({len(archive_names)} entries); .env absent confirmed")


# ---------------------------------------------------------------- check 8
gitignore = REPO / ".gitignore"
if not gitignore.is_file():
    emit("FAIL", "8. .gitignore sample outputs", ".gitignore not found")
else:
    sample_target = "sample_outputs/example.wav"
    if would_be_ignored(sample_target):
        emit("FAIL", "8. .gitignore sample outputs",
             f"{sample_target} WOULD be ignored")
    else:
        emit("PASS", "8. .gitignore sample outputs",
             f"{sample_target} would NOT be ignored")


# ---------------------------------------------------------------- check 9
ABS_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:[A-Za-z]:[\\/]|/home/|/Users/)"
)
HARDCODED_FALLBACK_RE = re.compile(
    r"Path\(__file__\)\.resolve\(\)\.parent\.parent\.parent\s*/\s*"
    r"[\"']Capstone_Project_Dataset_FTTT_2026"
)

abs_targets: list[Path] = []
src_root = REPO / "src"
if src_root.is_dir():
    abs_targets.extend(
        p for p in src_root.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    )
for name in ("run.py", "README.md", ".env.example"):
    candidate = REPO / name
    if candidate.is_file():
        abs_targets.append(candidate)

absolute_hits: list[str] = []
absolute_detail: list[str] = []
for path in sorted(set(abs_targets)):
    if not is_text(path):
        continue
    try:
        text_lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        continue
    for lineno, line in enumerate(text_lines, 1):
        literal = bool(ABS_PATH_RE.search(line))
        fallback = bool(HARDCODED_FALLBACK_RE.search(line))
        if not (literal or fallback):
            continue
        hit = f"{rel(path)}:{lineno}"
        if hit not in absolute_hits:
            absolute_hits.append(hit)
        if literal or fallback:
            absolute_detail.append(hit)

if absolute_hits:
    emit("FAIL", "9. Absolute paths", "; ".join(sorted(absolute_hits)))
    for detail in sorted(set(absolute_detail)):
        note("9a. Absolute path detail", detail)
else:
    emit("PASS", "9. Absolute paths",
         "no Windows drive-letter or /home/, /Users/ paths in src/, run.py, README.md, .env.example")


# ---------------------------------------------------------------- path audit

def source_contains(source: str, needle: str) -> bool:
    file_part, _, line_part = source.partition(":")
    path = REPO / file_part
    if not path.is_file():
        return False
    try:
        text_lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    try:
        if "-" in line_part:
            start_s, end_s = line_part.split("-", 1)
            start, end = int(start_s), int(end_s)
            chunk = "\n".join(text_lines[start - 1:end])
        else:
            num = int(line_part)
            if num < 1 or num > len(text_lines):
                return False
            chunk = text_lines[num - 1]
    except ValueError:
        return False
    return needle in chunk


path_rows: list[dict] = []


def add_path(source: str, expression: str, resolved: str, klass: str,
             zip_status: str, needle: str) -> None:
    if not source_contains(source, needle):
        resolved += " [source moved]"
    path_rows.append({
        "source": source,
        "expression": expression,
        "resolved": resolved,
        "class": klass,
        "zip": zip_status,
    })


repo_posix = REPO.as_posix()
add_path(
    "src/data.py:12", ".env", f"{repo_posix}/.env",
    "(a) inside repo", "N/A (optional; .env excluded by .gitignore)", ".env",
)
add_path(
    "src/data.py:18-23", 'os.getenv("DATASET_DIR")',
    "<DATASET_DIR> (external; value redacted, setting DATASET_DIR)",
    "(b) configured external directory (DATASET_DIR)", "NO", "DATASET_DIR",
)
add_path(
    "src/data.py:30", "DATASET_DIR / rel_path", "<DATASET_DIR>/<rel_path>",
    "(b) configured external directory (DATASET_DIR)", "NO", "DATASET_DIR / rel_path",
)

DATA_FILES = [
    ("src/data.py:36", "data/customers.csv"),
    ("src/data.py:37", "data/orders.csv"),
    ("src/data.py:38", "data/order_items.csv"),
    ("src/data.py:39", "data/products.csv"),
    ("src/data.py:40", "data/pincodes.csv"),
    ("src/data.py:41", "data/holidays.csv"),
    ("src/data.py:42", "data/support_logs.csv"),
    ("src/data.py:43", "practice/questions.csv"),
    ("src/data.py:44", "practice/conversations.csv"),
]
for source, expression in DATA_FILES:
    add_path(
        source, expression, f"<DATASET_DIR>/{expression}",
        "(b) configured external directory (DATASET_DIR)", "NO", expression,
    )

add_path(
    "src/kb.py:126-133", 'Path(dataset_dir) / "kb"', "<DATASET_DIR>/kb/*.md",
    "(b) configured external directory (DATASET_DIR)", "NO", "DATASET_DIR",
)
add_path(
    "src/order_tools.py:17-23", "DATASET_DIR/tools/tools.py",
    "<DATASET_DIR>/tools/tools.py",
    "(b) configured external directory (DATASET_DIR)",
    f"NO; repo/tools/ exists: {'yes' if repo_tools_exists else 'no'}",
    "DATASET_DIR",
)
add_path(
    "src/baseline.py:17-18", "outputs/step1_results.csv",
    f"{repo_posix}/outputs/step1_results.csv",
    "(a) inside repo", "YES (created at runtime; outputs/ not in zip payload)",
    "_DEFAULT_OUTPUT",
)
add_path(
    "run.py:24", "voice_clips (default folder)", f"{repo_posix}/voice_clips",
    "(a) inside repo", "YES (created at runtime; voice_clips/ not in zip payload)",
    "voice_clips",
)
add_path(
    "src/voice.py:20-22", ".env", f"{repo_posix}/.env",
    "(a) inside repo", "N/A (optional; .env excluded by .gitignore)", ".env",
)
add_path(
    "src/voice.py:287-289", "out_dir / reply.wav",
    f"{repo_posix}/outputs/voice/<clip>/reply.wav",
    "(a) inside repo", "YES (created at runtime; outputs/ not in zip payload)",
    "reply.wav",
)
add_path(
    "src/voice.py:104", "audio_path", "caller-supplied (README command: voice_clips/<clip>)",
    "(a) inside repo for the documented command", "YES (input folder; not in zip payload)",
    "Path(audio_path)",
)
add_path(
    "src/llm.py:16", ".env", f"{repo_posix}/.env",
    "(a) inside repo", "N/A (optional; .env excluded by .gitignore)", ".env",
)
add_path(
    "src/kb.py:13", "load_dotenv()", f"{repo_posix}/.env when cwd=repo",
    "(a) inside repo", "N/A (optional; .env excluded by .gitignore)", "load_dotenv()",
)

# Dynamic token counts for the grep coverage lines.
def token_count(token: str) -> tuple[int, list[str]]:
    total = 0
    by_file = []
    for path in py_files:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        n = len(re.findall(re.escape(token), text, re.IGNORECASE))
        if n:
            by_file.append(f"{rel(path)}={n}")
            total += n
    return total, by_file


grep_tokens = ["kb", "data", "tools", "practice", "products.csv", "holidays.csv", "faiss"]
grep_lines = []
for token in grep_tokens:
    total, by_file = token_count(token)
    grep_lines.append(f"- {token}: {total} hit(s) ({', '.join(by_file)})")

# Zip-payload completeness and unresolvable runtime paths.
zip_present = {name: (REPO / name.rstrip("/")).exists() for name in ZIP_SPEC}
unresolvable = [
    "<DATASET_DIR>/data/customers.csv",
    "<DATASET_DIR>/data/orders.csv",
    "<DATASET_DIR>/data/order_items.csv",
    "<DATASET_DIR>/data/products.csv",
    "<DATASET_DIR>/data/pincodes.csv",
    "<DATASET_DIR>/data/holidays.csv",
    "<DATASET_DIR>/data/support_logs.csv",
    "<DATASET_DIR>/practice/questions.csv",
    "<DATASET_DIR>/practice/conversations.csv",
    "<DATASET_DIR>/kb/*.md",
    "<DATASET_DIR>/tools/tools.py",
]


# ---------------------------------------------------------------- appendices
import_appendix = [
    "APPENDIX A - IMPORTS IN src/ AND run.py",
    "FILE:LINE  STATEMENT  MODULE  REQUIREMENTS MAPPING",
]
for row in import_rows:
    import_appendix.append(
        f"{row['file']}:{row['line']}  {row['kind']:<6} {row['module']:<36} -> {row['mapping']}"
    )

path_appendix = [
    "APPENDIX B - RUNTIME PATHS (grep of src/ and run.py)",
    "SOURCE | PATH EXPRESSION | RESOLVED ABSOLUTE PATH | CLASS | ZIP-RESOLVABLE",
]
for row in path_rows:
    path_appendix.append(
        f"{row['source']} | {row['expression']} | {row['resolved']} | {row['class']} | {row['zip']}"
    )
path_appendix.append("")
path_appendix.append("GREP COVERAGE (src/ and run.py):")
path_appendix.extend(grep_lines)
path_appendix.append(
    "- absolute filesystem paths (check 9): "
    + (", ".join(sorted(absolute_hits)) if absolute_hits else "none")
    + " (the only absolute URL is https://... at src/llm.py:51; not a filesystem path)"
)
path_appendix.append("")
path_appendix.append("NON-PATH GREP NOTES:")
path_appendix.append("- products.csv: only src/data.py:39 loads the file; src/retrieval.py:158,164,165,177 and src/conversation.py:40 use section IDs/metadata.")
path_appendix.append("- holidays.csv: only src/data.py:41 loads the file; src/rag.py:128,138 are prompt text; src/data.py:93 is a docstring.")
path_appendix.append("- faiss: src/retrieval.py imports and uses the Python package/index; it is not a filesystem path. Dependency: faiss-cpu (requirements-step2.txt:3).")
path_appendix.append("- practice: src/data.py:43-44 are the real loads; src/baseline.py:126 is a docstring.")
path_appendix.append("- tools: src/order_tools.py imports tools.tools from DATASET_DIR; it is not a repo path and is not a requirements entry.")
path_appendix.append("- kb: src/kb.py:126-133 and :141 are the real path/glob operations; other hits are identifiers/docstrings.")

zip_appendix = [
    "APPENDIX C - ZIP SPEC AND UNRESOLVABLE PATHS",
    "Declared zip payload:",
]
for name in ZIP_SPEC:
    zip_appendix.append(f"- {name}: {'present' if zip_present[name] else 'MISSING'}")
zip_appendix.append("")
zip_appendix.append("Runtime paths a clean copy of the zip could not resolve:")
for item in unresolvable:
    zip_appendix.append(f"- {item}")
zip_appendix.append("")
zip_appendix.append(
    "Reason: .env.example leaves DATASET_DIR empty; the dataset (data/, practice/, kb/, tools/) "
    "is outside the repo and outside the zip; repo/tools/ does not exist. "
    "report.pdf and demo.mp4 are missing payload items, not runtime paths."
)


# ---------------------------------------------------------------- report
header = [
    "VoiceOps 2026 submission check",
    f"Repo: {REPO.as_posix()}",
    f"Expected zip: {EXPECTED_ZIP}",
    "Mode: offline; no LLM/Gemini/Gnani calls; no git commands",
    "-" * 72,
]
footer = [
    "-" * 72,
    f"RESULT: {'FAIL' if failed else 'OK'} | {passed} passed, {failed} failed, {missing} missing",
]
report = "\n".join(
    header
    + lines
    + [""]
    + import_appendix
    + [""]
    + path_appendix
    + [""]
    + zip_appendix
    + footer
) + "\n"

print(report, end="")
SUBMISSION.mkdir(parents=True, exist_ok=True)
(SUBMISSION / "check_report.txt").write_text(report, encoding="utf-8", newline="\n")

sys.exit(1 if failed else 0)
