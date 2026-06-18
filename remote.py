"""
remote.py -- run the heavy CXR scan on the GPU box, keep the data-vis local.

The split this enables: the laptop holds the project and does ALL the data-vis +
PDF export (where matplotlib and the xelatex/webpdf toolchain are set up), while
the lab box (an RTX 5080, ssh host 'qlmc') only does the GPU-heavy Monte-Carlo
sweep. This script ships the current code up, runs scan.py there, and pulls the
resulting checkpoint back into ./checkpoints -- so you never hand-ssh in or copy
files, and you never need a PDF toolchain on the lab box.

    python remote.py scan mose2              # sync code up, run sweep, pull checkpoint
    python remote.py scan mose2 --quick       # tiny grid smoke test
    python remote.py scan mose2 --no-sync     # skip the code upload (code unchanged)
    python remote.py pull mose2               # just fetch an existing checkpoint
    python remote.py sync                      # only push the current code

Then locally: open analysis.ipynb (same MATERIAL) or run export_pdf.py.

Transport is ssh/scp only (uses the 'qlmc' host in ~/.ssh/config, cloudflared
ProxyCommand and all) -- no rsync dependency, so it works from Windows Git Bash.
Override the box via env: CXR_REMOTE_HOST / CXR_REMOTE_DIR / CXR_REMOTE_UV.
"""

import argparse
import io
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

HOST = os.environ.get("CXR_REMOTE_HOST", "qlmc")
REMOTE_DIR = os.environ.get("CXR_REMOTE_DIR", "/home/aamador/dev/cxr_model")
REMOTE_UV = os.environ.get("CXR_REMOTE_UV", "/home/aamador/.local/bin/uv")
LOCAL_ROOT = Path(__file__).resolve().parent

# what `sync` ships up: the code that changes, not data/ (static, already there)
# or checkpoints/ (the output we pull back the other way).
SYNC_PATHS = ["src", "scan.py", "pyproject.toml"]

# text extensions whose CRLF is normalized to LF before tarring (see _add_to_tar):
# the laptop is Windows so its working files are CRLF, and shipping those over the
# box's LF checkout dirties `git status` there even though content is identical.
TEXT_EXTS = {".py", ".toml", ".cfg", ".ini", ".txt", ".md", ".csv"}


def _run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _add_to_tar(tar, local, arcname):
    """Add one local file to the tar. Text files (TEXT_EXTS) get CRLF->LF so the
    box receives LF-clean content -- no cosmetic `git status` diff there, so a
    later `git pull` isn't blocked. Other files are added verbatim."""
    if local.suffix.lower() in TEXT_EXTS:
        data = local.read_bytes().replace(b"\r\n", b"\n")
        info = tarfile.TarInfo(name=arcname)
        info.size = len(data)
        info.mtime = int(local.stat().st_mtime)
        info.mode = 0o644
        tar.addfile(info, io.BytesIO(data))
    else:
        tar.add(local, arcname=arcname)


def sync_code():
    """Tar SYNC_PATHS up (CRLF->LF normalized for text, via _add_to_tar) and
    extract them over the repo on the box.

    Normalizing line endings here keeps the edit-locally / run-remotely loop --
    it ships the current WORKING tree (no commit required) yet stays LF-clean, so
    the box's `git status` doesn't flag every synced .py as modified and a later
    `git pull` there isn't blocked. (The committed-state alternative -- `git push`
    from the laptop + `git fetch && git reset --hard origin/<branch>` on the box --
    would force a commit before every run, which this loop is built to avoid.)"""
    with tempfile.TemporaryDirectory() as td:
        tarpath = os.path.join(td, "cxr_code.tgz")
        with tarfile.open(tarpath, "w:gz") as t:
            for p in SYNC_PATHS:
                local = LOCAL_ROOT / p
                if not local.exists():
                    continue
                if local.is_dir():
                    for f in sorted(local.rglob("*")):
                        if f.is_file():
                            arc = (Path(p) / f.relative_to(local)).as_posix()
                            _add_to_tar(t, f, arc)
                else:
                    _add_to_tar(t, local, p)
        _run(["scp", tarpath, f"{HOST}:/tmp/cxr_code.tgz"])
    _run(
        [
            "ssh",
            HOST,
            f"cd {REMOTE_DIR} && tar xzf /tmp/cxr_code.tgz && rm -f /tmp/cxr_code.tgz",
        ]
    )


def remote_scan(material, quick=False, workers=None):
    cmd = f"cd {REMOTE_DIR} && {REMOTE_UV} run --no-sync python scan.py {material}"
    if quick:
        cmd += " --quick"
    if workers is not None:
        cmd += f" --workers {workers}"
    _run(["ssh", HOST, cmd])


def pull(stem):
    """Fetch checkpoints/<stem>.pkl back from the box (stem = material, or
    material_quick for a --quick run)."""
    dest = LOCAL_ROOT / "checkpoints"
    dest.mkdir(exist_ok=True)
    _run(
        [
            "scp",
            f"{HOST}:{REMOTE_DIR}/checkpoints/{stem}.pkl",
            str(dest / f"{stem}.pkl"),
        ]
    )
    print(f"pulled -> checkpoints/{stem}.pkl")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="sync code, run the sweep on the box, pull checkpoint")
    s.add_argument("material")
    s.add_argument("--quick", action="store_true")
    s.add_argument("--workers", type=int, default=None)
    s.add_argument("--no-sync", action="store_true", help="skip the code upload")

    p = sub.add_parser("pull", help="fetch an existing checkpoint from the box")
    p.add_argument("material")

    sub.add_parser("sync", help="push the current code to the box only")

    args = ap.parse_args()
    if args.cmd == "sync":
        sync_code()
    elif args.cmd == "pull":
        pull(args.material)
    elif args.cmd == "scan":
        if not args.no_sync:
            sync_code()
        remote_scan(args.material, args.quick, args.workers)
        stem = f"{args.material}_quick" if args.quick else args.material
        pull(stem)
        print(
            f"\ndone. checkpoints/{stem}.pkl is local; open analysis.ipynb with "
            f"MATERIAL='{stem}' (or run export_pdf.py) -- all viz/PDF stays local."
        )


if __name__ == "__main__":
    main()
