"""Build the Agite LiteLLM wheel from upstream's own published wheel.

Upstream ships platform wheels with an optional Rust extension; LiteLLM runs
without it (`litellm.rust_bridge.loader` returns None). Agite installs one
pure-Python wheel on every platform, so this takes upstream's wheel for the
base version, applies this branch's source changes under `litellm/`, drops the
native module, and writes `litellm-<version>-py3-none-any.whl`.

usage: python agite/build_wheel.py <upstream.whl> <upstream sha256> <base tag> <out dir>
"""

from __future__ import annotations

import base64
import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from pathlib import Path


def _digest(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def main(upstream: str, sha256: str, base_tag: str, out_dir: str) -> Path:
    repo = Path(__file__).resolve().parents[1]
    data = Path(upstream).read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise SystemExit("upstream wheel digest does not match")
    project = tomllib.loads((repo / "pyproject.toml").read_text())["project"]
    version = project["version"]
    base = base_tag.removeprefix("v")
    if not version.startswith(base + "+agite."):
        raise SystemExit(f"version {version} is not an Agite build of {base}")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "wheel"
        with zipfile.ZipFile(upstream) as wheel:
            wheel.extractall(root)
        for native in root.glob("litellm/rust_bridge/_native*"):
            native.unlink()
        patch = subprocess.run(
            ["git", "diff", base_tag, "--", "litellm/"],
            cwd=repo, check=True, capture_output=True,
        ).stdout
        if patch:
            subprocess.run(["patch", "-p1", "--forward", "--quiet"], cwd=root, input=patch, check=True)
        for leftover in list(root.rglob("*.orig")) + list(root.rglob("*.rej")):
            raise SystemExit(f"patch left {leftover.name}")
        (old_info,) = root.glob("litellm-*.dist-info")
        info = root / f"litellm-{version}.dist-info"
        old_info.rename(info)
        metadata = (info / "METADATA").read_text()
        metadata = re.sub(r"^Version: .*$", f"Version: {version}", metadata, count=1, flags=re.M)
        extra = [d for d in project["dependencies"] if d.split(">")[0].split("=")[0].split("<")[0].strip() == "filelock"]
        for requirement in extra:
            if f"Requires-Dist: {requirement}" not in metadata:
                metadata = metadata.replace("Requires-Dist:", f"Requires-Dist: {requirement}\nRequires-Dist:", 1)
        (info / "METADATA").write_text(metadata)
        wheel_text = (info / "WHEEL").read_text()
        wheel_text = re.sub(r"^Root-Is-Purelib: .*$", "Root-Is-Purelib: true", wheel_text, flags=re.M)
        wheel_text = re.sub(r"^Tag: .*\n", "", wheel_text, flags=re.M).rstrip("\n") + "\nTag: py3-none-any\n"
        (info / "WHEEL").write_text(wheel_text)
        (info / "AGITE_FORK.md").write_bytes((repo / "AGITE_FORK.md").read_bytes())
        record = info / "RECORD"
        rows = []
        for path in sorted(p for p in root.rglob("*") if p.is_file() and p != record):
            relative = path.relative_to(root).as_posix()
            content = path.read_bytes()
            rows.append(f"{relative},{_digest(content)},{len(content)}")
        rows.append(f"{record.relative_to(root).as_posix()},,")
        record.write_text("\n".join(rows) + "\n")
        out = Path(out_dir) / f"litellm-{version}-py3-none-any.whl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as wheel:
            for path in sorted(p for p in root.rglob("*") if p.is_file()):
                info_entry = zipfile.ZipInfo(path.relative_to(root).as_posix(), (1980, 1, 1, 0, 0, 0))
                info_entry.external_attr = 0o644 << 16
                info_entry.compress_type = zipfile.ZIP_DEFLATED
                wheel.writestr(info_entry, path.read_bytes())
    print(out, hashlib.sha256(out.read_bytes()).hexdigest())
    return out


if __name__ == "__main__":
    main(*sys.argv[1:])
