"""Rebuild a working HuggingFace cache from a macOS-copied one.

WHY THIS EXISTS
===============
The HuggingFace hub cache stores each file once in ``blobs/<sha>`` and points at
it from ``snapshots/<revision>/<filename>`` with a **symlink**. When that tree is
copied off a Mac over SMB (or onto any filesystem that cannot represent a
symlink), macOS writes an ``XSym`` placeholder instead: a 1067-byte regular file
whose body is the link target in plain text::

    XSym\\n
    0076\\n                     <- target length
    <md5 of target>\\n
    ../../blobs/<sha>\\n
    <space padding to 1067 bytes>

The blobs themselves arrive intact, so nothing is lost — the links just need
rebuilding. This script parses every ``XSym`` placeholder and materialises the
real file, producing a cache ``transformers`` and ``timm`` can load offline.

Hardlinks are used where possible so the blob is not duplicated on disk; the
script falls back to a copy across volumes or on any filesystem that refuses.

USAGE
=====
    python repair_hf_cache.py <source-cache> [--dest <dir>] [--dry-run]

Default destination is the standard hub cache (``~/.cache/huggingface/hub``),
which is deliberately *not* inside OneDrive — multi-gigabyte weights should not
be sync targets.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

XSYM_MAGIC = b"XSym\n"
XSYM_SIZE = 1067


def default_cache() -> Path:
    """The hub cache HuggingFace itself would use, honouring HF_HOME."""
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def read_xsym_target(path: Path) -> str | None:
    """The link target inside an XSym placeholder, or None if not one."""
    try:
        with path.open("rb") as fh:
            head = fh.read(XSYM_SIZE)
    except OSError:
        return None
    if not head.startswith(XSYM_MAGIC):
        return None
    parts = head.split(b"\n")
    # XSym / length / md5 / target
    if len(parts) < 4:
        return None
    target = parts[3].decode("utf-8", "replace").strip()
    return target or None


def link_or_copy(source: Path, dest: Path) -> str:
    """Hardlink *source* to *dest*, falling back to a copy.  Returns the mode."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    try:
        os.link(source, dest)
        return "hardlink"
    except OSError:
        shutil.copy2(source, dest)
        return "copy"


def human(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}TB"


def repair(source_root: Path, dest_root: Path, dry_run: bool = False) -> int:
    """Copy every ``models--*`` tree from *source_root* into *dest_root*, fixed."""
    model_dirs = sorted(d for d in source_root.iterdir()
                        if d.is_dir() and d.name.startswith("models--"))
    if not model_dirs:
        print(f"No models--* directories under {source_root}", file=sys.stderr)
        return 1

    problems = 0
    for model_dir in model_dirs:
        dest_model = dest_root / model_dir.name
        print(f"\n=== {model_dir.name}")

        # 1. Blobs: the real payload.
        blobs_src = model_dir / "blobs"
        total = 0
        if blobs_src.is_dir():
            for blob in sorted(blobs_src.iterdir()):
                if not blob.is_file():
                    continue
                size = blob.stat().st_size
                total += size
                dest_blob = dest_model / "blobs" / blob.name
                if dry_run:
                    print(f"  blob     {human(size):>9}  {blob.name[:16]}…")
                    continue
                if dest_blob.exists() and dest_blob.stat().st_size == size:
                    mode = "exists"
                else:
                    mode = link_or_copy(blob, dest_blob)
                print(f"  blob     {human(size):>9}  {blob.name[:16]}…  [{mode}]")

        # 2. Snapshots: rebuild XSym placeholders, copy anything already real.
        for snap in sorted((model_dir / "snapshots").glob("*")):
            if not snap.is_dir():
                continue
            for entry in sorted(snap.rglob("*")):
                if not entry.is_file():
                    continue
                relative = entry.relative_to(model_dir)
                dest_entry = dest_model / relative
                target = read_xsym_target(entry)

                if target is None:
                    # A genuinely real file (small configs sometimes survive).
                    if dry_run:
                        print(f"  file     {human(entry.stat().st_size):>9}  {relative}")
                    else:
                        mode = link_or_copy(entry, dest_entry)
                        print(f"  file     {human(entry.stat().st_size):>9}  "
                              f"{relative}  [{mode}]")
                    continue

                resolved = (entry.parent / target).resolve()
                if not resolved.is_file():
                    print(f"  MISSING  target for {relative} -> {target}", file=sys.stderr)
                    problems += 1
                    continue
                if dry_run:
                    print(f"  XSym     {human(resolved.stat().st_size):>9}  "
                          f"{relative} -> blobs/{resolved.name[:16]}…")
                    continue
                # Point at the blob we just placed in the destination.
                dest_blob = dest_model / "blobs" / resolved.name
                source_for_link = dest_blob if dest_blob.is_file() else resolved
                mode = link_or_copy(source_for_link, dest_entry)
                print(f"  repaired {human(resolved.stat().st_size):>9}  "
                      f"{relative}  [{mode}]")

        # 3. refs + .no_exist: small bookkeeping HF consults.
        for extra in ("refs", ".no_exist"):
            src_extra = model_dir / extra
            if not src_extra.is_dir():
                continue
            for item in src_extra.rglob("*"):
                if item.is_file() and not dry_run:
                    link_or_copy(item, dest_model / item.relative_to(model_dir))

        print(f"  total blobs: {human(total)}")

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source", type=Path,
                        help="the copied hf-cache directory containing models--*")
    parser.add_argument("--dest", type=Path, default=None,
                        help=f"destination hub cache (default: {default_cache()})")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would happen, change nothing")
    args = parser.parse_args(argv)

    source = args.source.expanduser().resolve()
    dest = (args.dest or default_cache()).expanduser()
    if not source.is_dir():
        print(f"Source not found: {source}", file=sys.stderr)
        return 1

    print(f"source: {source}")
    print(f"dest  : {dest}")
    if args.dry_run:
        print("(dry run — nothing will be written)")
    else:
        dest.mkdir(parents=True, exist_ok=True)

    problems = repair(source, dest, args.dry_run)
    if problems:
        print(f"\n{problems} unresolved link(s) — cache is incomplete.", file=sys.stderr)
        return 1
    if not args.dry_run:
        print(f"\nDone. Point HuggingFace at it with:  set HF_HOME={dest.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
