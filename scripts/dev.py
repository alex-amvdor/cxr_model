#!/usr/bin/env python3
"""Small cross-platform developer command runner for cxr_model.

This keeps Claude and humans out of shell one-liner hell on Windows.
Run via:

    uv run python scripts/dev.py <command>

Commands:
    repo-map   print a compact repo tree and the canonical commands
    lint       run Ruff over source, tests, dev helpers, and checks
    format     run Ruff formatter
    nbqa       lint notebooks with nbQA + Ruff
    nbstrip    strip notebook outputs in-place
    test       run the fast test suite
    verify     lint + test
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_SKIP_PARTS = {
    ".venv",
    ".git",
    "checkpoints",
    "results",
    "docs/_build",
    "docs/_autosummary",
}


def run(*args: str, cwd: Path = ROOT) -> None:
    subprocess.run([sys.executable, *args], cwd=cwd, check=True)


def iter_notebooks() -> list[Path]:
    notebooks: list[Path] = []
    for path in ROOT.rglob("*.ipynb"):
        rel = path.relative_to(ROOT).as_posix()
        if any(skip in rel for skip in NOTEBOOK_SKIP_PARTS):
            continue
        notebooks.append(path)
    return sorted(notebooks)


def cmd_repo_map(_: argparse.Namespace) -> None:
    interesting = [
        "src/cxr_model",
        "tests",
        "checks",
        "docs",
        "dev",
        ".claude",
        "scripts",
    ]
    print("cxr_model repo map")
    for rel in interesting:
        path = ROOT / rel
        if not path.exists():
            continue
        if path.is_dir():
            print(f"{rel}/")
            entries = []
            for child in sorted(path.iterdir()):
                if child.name.startswith("."):
                    continue
                if child.is_dir():
                    entries.append(f"  {child.name}/")
                else:
                    entries.append(f"  {child.name}")
            for line in entries[:40]:
                print(line)
            if len(entries) > 40:
                print("  ...")
            print()
        else:
            print(rel)
    print("Canonical commands:")
    for line in [
        "uv run python scripts/dev.py lint",
        "uv run python scripts/dev.py format",
        "uv run python scripts/dev.py test",
        "uv run python scripts/dev.py verify",
        "uv run python scripts/dev.py nbqa",
        "uv run python scripts/dev.py nbstrip",
    ]:
        print(f"  {line}")


def cmd_lint(_: argparse.Namespace) -> None:
    run("-m", "ruff", "check", "src", "tests", "dev", "checks")


def cmd_format(_: argparse.Namespace) -> None:
    run("-m", "ruff", "format", "src", "tests", "dev", "checks")


def cmd_nbqa(_: argparse.Namespace) -> None:
    notebooks = iter_notebooks()
    if not notebooks:
        print("No notebooks found.")
        return
    run("-m", "nbqa", "ruff", "check", *[str(p) for p in notebooks])


def cmd_nbstrip(_: argparse.Namespace) -> None:
    notebooks = iter_notebooks()
    if not notebooks:
        print("No notebooks found.")
        return
    run("-m", "nbstripout", *[str(p) for p in notebooks])


def cmd_test(_: argparse.Namespace) -> None:
    run("-m", "pytest", "-q")


def cmd_verify(_: argparse.Namespace) -> None:
    cmd_lint(argparse.Namespace())
    cmd_test(argparse.Namespace())


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="dev.py")
    sub = ap.add_subparsers(dest="command", required=True)

    for name, fn in [
        ("repo-map", cmd_repo_map),
        ("lint", cmd_lint),
        ("format", cmd_format),
        ("nbqa", cmd_nbqa),
        ("nbstrip", cmd_nbstrip),
        ("test", cmd_test),
        ("verify", cmd_verify),
    ]:
        sp = sub.add_parser(name)
        sp.set_defaults(func=fn)
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
