"""Package PathLearn for a move to another Windows desktop, without a remote.

    python tools/transfer.py export --dest E:\\PathLearn-Transfer --all
    python tools/transfer.py verify --bundle E:\\PathLearn-Transfer

Copy the bundle to the other machine (USB, network share, external drive) and
run ``install.ps1`` inside it.

WHAT MOVES, AND WHAT MUST NOT
=============================
* **Source** (~5 MB) — always.
* **Extractors** (~4 GB) — optional but usually worth it. Re-converting needs
  HuggingFace access approval again, per account, and is not instant.
* **Banks** — the patch bank and geometry bank are your work, not source.
  SQLite is checkpointed before copying so the ``-wal`` sidecar is not needed.
* **Wheels** — optional. Makes the far side installable with no internet.
* **The venv is never copied.** It holds absolute paths — the editable install
  records ``c:\\users\\<you>\\...`` inside a path finder — so a copied venv
  imports the *source that is no longer there*. It is rebuilt on arrival.
* **Slides are not copied.** They are hundreds of gigabytes and usually
  already present, or moved separately. Annotation sidecars travel with them,
  since they live beside the slide.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "pathlearn"

#: Never copied into a bundle: build noise, caches, and anything machine-local.
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".venv", "venv", ".git",
             "node_modules", ".mypy_cache", ".ruff_cache"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".log", ".tmp"}
SKIP_NAMES = {"pathlearn-run.log", "Thumbs.db", ".DS_Store"}

APP_DATA = Path.home() / "AppData" / "Local" / "PathLearn"
BANK_FILES = ("bank.db", "geometry_bank.json", "extractor-equivalences.json")

MANIFEST_NAME = "MANIFEST.json"


@dataclass
class Report:
    files: int = 0
    bytes: int = 0
    skipped: list[str] = field(default_factory=list)

    def add(self, path: Path) -> None:
        self.files += 1
        self.bytes += path.stat().st_size

    @property
    def human(self) -> str:
        return _human(self.bytes)


def _human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size:.0f} B"
        size /= 1024
    return f"{size:.1f} GB"


def _wanted(path: Path) -> bool:
    if path.name in SKIP_NAMES or path.suffix in SKIP_SUFFIXES:
        return False
    return not any(part in SKIP_DIRS for part in path.parts)


def copy_tree(source: Path, destination: Path, report: Report) -> None:
    """Copy *source* into *destination*, skipping build noise."""
    for item in sorted(source.rglob("*")):
        if item.is_dir() or not _wanted(item.relative_to(source)):
            continue
        target = destination / item.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
        report.add(target)


#: Every file belonging to one extractor shares its stem — the ONNX graph, its
#: external weight sidecar, and the descriptor.
def _extractor_name(path: Path) -> str:
    """`uni2-h.onnx.data` and `uni2-h.pathlearn-extractor.json` -> `uni2-h`."""
    name = path.name
    for suffix in (".pathlearn-extractor.json", ".paninextractor.json",
                   ".onnx.data", ".onnx"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def installed_extractors(folder: Path) -> set[str]:
    return {_extractor_name(f) for f in folder.iterdir() if f.is_file()}


def select_extractors(folder: Path, wanted: set[str] | None) -> list[Path]:
    """Every file belonging to the chosen extractors.

    Selecting by name rather than by file matters for UNI2-h: its weights live
    in a separate `uni2-h.onnx.data` because they exceed protobuf's 2 GB limit,
    and a bundle carrying the graph without the sidecar fails at load time with
    an unhelpful error.
    """
    files = [f for f in sorted(folder.iterdir()) if f.is_file()]
    if wanted is None:
        return files
    return [f for f in files if _extractor_name(f) in wanted]


def checkpoint_sqlite(path: Path) -> None:
    """Fold the write-ahead log back into the database before copying it.

    Without this the bundle needs ``bank.db-wal`` too, and a copy taken while
    the app is running can otherwise miss recent patches entirely.
    """
    if not path.exists():
        return
    try:
        connection = sqlite3.connect(str(path))
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.close()
    except sqlite3.Error as exc:
        print(f"  ! could not checkpoint {path.name}: {exc}")


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(bundle: Path, notes: dict) -> Path:
    """Record every file with its size and hash, so the copy can be checked."""
    entries = []
    for item in sorted(bundle.rglob("*")):
        if item.is_dir() or item.name == MANIFEST_NAME:
            continue
        entries.append({
            "path": item.relative_to(bundle).as_posix(),
            "bytes": item.stat().st_size,
            "sha256": sha256(item),
        })
    manifest = {"created_on": os.environ.get("COMPUTERNAME", "unknown"),
                "python": sys.version.split()[0],
                "files": entries,
                "totalBytes": sum(e["bytes"] for e in entries),
                **notes}
    path = bundle / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def freeze(interpreter: Path) -> str:
    """Pin what this machine actually runs, minus the editable install itself."""
    try:
        done = subprocess.run([str(interpreter), "-m", "pip", "freeze"],
                              capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"# pip freeze failed: {exc}\n"
    lines = [line for line in done.stdout.splitlines()
             if line.strip() and not line.startswith("-e ")
             and "pathlearn" not in line.lower()]
    return "\n".join(lines) + "\n"


def download_wheels(interpreter: Path, requirements: Path, into: Path) -> bool:
    """Pre-fetch every dependency so the far side needs no internet."""
    into.mkdir(parents=True, exist_ok=True)
    command = [str(interpreter), "-m", "pip", "download", "-r",
               str(requirements), "-d", str(into)]
    print(f"  downloading wheels into {into} (this takes a while)…")
    done = subprocess.run(command)
    return done.returncode == 0


# -- export ------------------------------------------------------------------

def export(dest: Path, *, extractors, banks: bool, wheels: bool,
           interpreter: Path) -> int:
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        print(f"Refusing to write into {dest}: it is not empty.")
        return 2
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Bundling into {dest}")
    report = Report()

    print("  source…")
    copy_tree(REPO_ROOT, dest / "PathLearn", report)

    requirements = dest / "requirements-lock.txt"
    requirements.write_text(freeze(interpreter), encoding="utf-8")
    report.add(requirements)
    print(f"  pinned {len(requirements.read_text(encoding='utf-8').splitlines())} "
          f"packages")

    if wheels:
        if download_wheels(interpreter, requirements, dest / "wheels"):
            for item in (dest / "wheels").iterdir():
                report.add(item)
        else:
            print("  ! wheel download failed; the far side will need internet")

    if extractors:
        source = APP_DATA / "Extractors"
        if not source.is_dir():
            print("  ! no extractors installed; skipping")
        else:
            wanted = None if extractors is True else set(extractors)
            chosen = select_extractors(source, wanted)
            if not chosen:
                print(f"  ! no extractor matched {sorted(wanted or [])}; "
                      f"installed: {sorted(installed_extractors(source))}")
            total = sum(f.stat().st_size for f in chosen)
            if chosen:
                print(f"  extractors ({_human(total)}): "
                      f"{sorted({_extractor_name(f) for f in chosen})}")
            target = dest / "Extractors"
            target.mkdir(parents=True, exist_ok=True)
            for item in chosen:
                shutil.copy2(item, target / item.name)
                report.add(target / item.name)

    if banks:
        target = dest / "AppData"
        target.mkdir(exist_ok=True)
        checkpoint_sqlite(APP_DATA / "bank.db")
        for name in BANK_FILES:
            source = APP_DATA / name
            if source.exists():
                shutil.copy2(source, target / name)
                report.add(target / name)
                print(f"  {name} ({_human(source.stat().st_size)})")

    (dest / "install.ps1").write_text(INSTALL_PS1, encoding="utf-8")
    (dest / "TRANSFER.md").write_text(TRANSFER_MD, encoding="utf-8")
    report.add(dest / "install.ps1")
    report.add(dest / "TRANSFER.md")

    print("  writing manifest (hashing everything)…")
    write_manifest(dest, {"includesExtractors": bool(extractors),
                          "includesBanks": bool(banks),
                          "includesWheels": bool(wheels)})

    print(f"\nDone: {report.files} files, {report.human}.")
    print(f"Copy {dest} to the other machine, then run install.ps1 inside it.")
    return 0


# -- verify ------------------------------------------------------------------

def verify(bundle: Path) -> int:
    """Re-hash the bundle and report anything that did not survive the copy."""
    bundle = Path(bundle)
    manifest_path = bundle / MANIFEST_NAME
    if not manifest_path.is_file():
        print(f"No {MANIFEST_NAME} in {bundle}.")
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    missing, corrupt, checked = [], [], 0
    for entry in manifest["files"]:
        path = bundle / entry["path"]
        if not path.is_file():
            missing.append(entry["path"])
            continue
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            corrupt.append(entry["path"])
        checked += 1

    print(f"Checked {checked} of {len(manifest['files'])} files "
          f"({_human(manifest['totalBytes'])}).")
    for name in missing:
        print(f"  MISSING  {name}")
    for name in corrupt:
        print(f"  CHANGED  {name}")
    if missing or corrupt:
        print("\nThe copy is not intact. Copy it again — a truncated .onnx "
              "fails only when a model is first used, which is a long way from "
              "here.")
        return 1
    print("Bundle is intact.")
    return 0


INSTALL_PS1 = r"""# PathLearn - install on this machine.
#   Right-click > Run with PowerShell, or:
#   powershell -ExecutionPolicy Bypass -File install.ps1
#
# Unattended, or into somewhere other than the defaults:
#   powershell -ExecutionPolicy Bypass -File install.ps1 -Target D:\PathLearn -Yes
param(
    [string]$Target,        # where the source goes
    [string]$Venv,          # where the virtual environment goes
    [switch]$Yes,           # never prompt; accept the defaults
    [switch]$SkipTests      # skip the post-install test run
)
$ErrorActionPreference = "Stop"
$bundle = Split-Path -Parent $MyInvocation.MyCommand.Path
$source = Join-Path $bundle "PathLearn"
if (-not $Venv) { $Venv = Join-Path $env:USERPROFILE ".venvs\pathlearn" }
$venv = $Venv
$appdata = Join-Path $env:LOCALAPPDATA "PathLearn"

Write-Host "PathLearn transfer installer" -ForegroundColor Cyan
Write-Host "  bundle : $bundle"

# 1. Python -------------------------------------------------------------------
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { throw "Python is not on PATH. Install Python 3.12+ from python.org, ticking 'Add to PATH'." }
$version = (& python -c "import sys; print('%d.%d' % sys.version_info[:2])")
Write-Host "  python : $version"
if ([version]$version -lt [version]"3.12") { throw "Python 3.12 or newer is required; found $version." }

# 2. Where the source will live ----------------------------------------------
$default = Join-Path $env:USERPROFILE "Documents\Projects\PathLearn"
if ($Target) {
    $target = $Target
} elseif ($Yes) {
    $target = $default
} else {
    $target = Read-Host "Install the source to [$default]"
    if ([string]::IsNullOrWhiteSpace($target)) { $target = $default }
}
if ((Test-Path $target) -and -not $Yes) {
    $answer = Read-Host "$target exists. Overwrite its contents? (y/N)"
    if ($answer -ne "y") { throw "Stopped: pick another location." }
}
New-Item -ItemType Directory -Force -Path $target | Out-Null
Write-Host "  copying source to $target"
Copy-Item -Path (Join-Path $source "*") -Destination $target -Recurse -Force

# 3. Virtual environment ------------------------------------------------------
# Never copied from the other machine: an editable install records absolute
# paths, so a copied venv imports source that is not there.
if (-not (Test-Path $venv)) {
    Write-Host "  creating venv at $venv"
    & python -m venv $venv
}
$vpy = Join-Path $venv "Scripts\python.exe"
& $vpy -m pip install --upgrade pip --quiet

$wheels = Join-Path $bundle "wheels"
$lock = Join-Path $bundle "requirements-lock.txt"
if (Test-Path $wheels) {
    Write-Host "  installing dependencies from bundled wheels (offline)"
    & $vpy -m pip install --no-index --find-links $wheels -r $lock
} else {
    Write-Host "  installing dependencies from PyPI (needs internet)"
    & $vpy -m pip install -r $lock
}

Write-Host "  installing PathLearn (editable)"
& $vpy -m pip install -e (Join-Path $target "pathlearn") --no-deps

# 4. Extractors and banks -----------------------------------------------------
New-Item -ItemType Directory -Force -Path $appdata | Out-Null
$bundledExtractors = Join-Path $bundle "Extractors"
if (Test-Path $bundledExtractors) {
    $dest = Join-Path $appdata "Extractors"
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    Write-Host "  copying extractors (this is the big one)"
    Copy-Item -Path (Join-Path $bundledExtractors "*") -Destination $dest -Recurse -Force
}
$bundledData = Join-Path $bundle "AppData"
if (Test-Path $bundledData) {
    Write-Host "  copying banks"
    Copy-Item -Path (Join-Path $bundledData "*") -Destination $appdata -Force
}

# 5. Prove it works -----------------------------------------------------------
$code = 0
if (-not $SkipTests) {
    Write-Host "`nRunning the test suite..." -ForegroundColor Cyan
    Push-Location (Join-Path $target "pathlearn")
    $env:QT_QPA_PLATFORM = "offscreen"
    & $vpy -m pytest -q
    $code = $LASTEXITCODE
    Remove-Item Env:\QT_QPA_PLATFORM
    Pop-Location
}

if ($code -ne 0) {
    Write-Host "`nTests failed. The install is present but something is wrong - do not trust results until this is resolved." -ForegroundColor Red
    exit $code
}

Write-Host "`nInstalled." -ForegroundColor Green
Write-Host "Launch with:  $venv\Scripts\pathlearn.exe"
Write-Host "Check Machine Learning > Installed Extractors to confirm the models came across."
"""


TRANSFER_MD = r"""# PathLearn transfer bundle

Copy this whole folder to the other Windows machine, then run `install.ps1`
inside it:

```
powershell -ExecutionPolicy Bypass -File install.ps1
```

It will ask where to put the source, build a fresh virtual environment,
install the dependencies, restore any extractors and banks in the bundle, and
run the test suite before telling you it worked.

## Check the copy first

USB copies of multi-gigabyte files do fail, and a truncated `.onnx` only shows
up much later, when a model is first used. From the installed source:

```
python tools/transfer.py verify --bundle <this folder>
```

## What is here

| Folder | What it is |
|---|---|
| `PathLearn/` | The source tree |
| `requirements-lock.txt` | Exact package versions from the source machine |
| `wheels/` | Present only if exported with `--wheels`; makes install work offline |
| `Extractors/` | ONNX models, if exported. They go to `%LOCALAPPDATA%\PathLearn\Extractors` |
| `AppData/` | Patch bank and geometry bank, if exported |

## What is deliberately not here

**The virtual environment.** It contains absolute paths — an editable install
records the source location inside a path finder — so a copied venv would try
to import source that does not exist on the new machine. `install.ps1` builds
a fresh one.

**Slides.** They are far too large, and their annotation sidecars live beside
them, so moving the slides moves the annotations with them.

**Model weights are licensed, not yours to redistribute.** Copying your own
converted extractors to your own second machine is fine. Passing them to
someone else is not — they must accept the licences and convert their own.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    export_parser = sub.add_parser("export", help="build a transfer bundle")
    export_parser.add_argument("--dest", type=Path, required=True,
                               help="empty folder to fill, e.g. a USB drive")
    export_parser.add_argument("--extractors", nargs="*", metavar="NAME",
                               help="include ONNX models; name them "
                                    "(e.g. --extractors uni2-h) or pass the "
                                    "flag alone for all of them")
    export_parser.add_argument("--banks", action="store_true",
                               help="include the patch and geometry banks")
    export_parser.add_argument("--wheels", action="store_true",
                               help="pre-download dependencies for an offline install")
    export_parser.add_argument("--all", action="store_true",
                               help="same as --extractors --banks --wheels")
    export_parser.add_argument("--python", type=Path, default=Path(sys.executable),
                               help="interpreter to read versions from")

    verify_parser = sub.add_parser("verify", help="check a bundle against its manifest")
    verify_parser.add_argument("--bundle", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "verify":
        return verify(args.bundle)
    # nargs="*" gives None when absent, [] when bare, names when listed.
    if args.extractors is None:
        extractors = True if args.all else False
    else:
        extractors = set(args.extractors) if args.extractors else True
    return export(args.dest,
                  extractors=extractors,
                  banks=args.banks or args.all,
                  wheels=args.wheels or args.all,
                  interpreter=args.python)


if __name__ == "__main__":
    raise SystemExit(main())
