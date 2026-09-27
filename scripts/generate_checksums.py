"""Generate SHA-256 checksums for versioned, non-ignored release files."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "SHA256SUMS.txt"
EXCLUDED_PARTS = {".git", ".venv", "__pycache__", "raw"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".parquet", ".npz", ".pkl", ".log"}


def include(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if path == OUTPUT or any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    return path.is_file()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    paths = sorted(path for path in ROOT.rglob("*") if include(path))
    lines = [f"{digest(path)}  {path.relative_to(ROOT).as_posix()}" for path in paths]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} checksums to {OUTPUT.name}")


if __name__ == "__main__":
    main()
