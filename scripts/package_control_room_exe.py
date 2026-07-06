"""Optional PyInstaller wrapper for the V5 control room.

The presentation-grade release is supported by the batch/PowerShell launchers.
This helper is intentionally optional: if PyInstaller is not already installed,
it reports the missing dependency and exits without changing protected artifacts.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_command(root: Path) -> list[str]:
    target = root / "scripts" / "control_room_server.py"
    release_dir = root / "reports" / "demo_control_room_v5"
    return [
        sys.executable,
        "-m",
        "PyInstaller",
        "--onefile",
        "--name",
        "control_room_v5",
        "--distpath",
        str(release_dir / "dist"),
        "--workpath",
        str(release_dir / "build"),
        "--specpath",
        str(release_dir),
        str(target),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Package the V5 control room as an optional exe.")
    parser.add_argument("--dry-run", action="store_true", help="Print the PyInstaller command without running it.")
    args = parser.parse_args(argv)

    root = repo_root()
    log_dir = root / "reports" / "demo_control_room_v5" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "package_control_room_exe.log"

    command = build_command(root)
    if args.dry_run:
        print(" ".join(command))
        log_path.write_text("DRY_RUN " + " ".join(command) + "\n", encoding="utf-8")
        return 0

    if importlib.util.find_spec("PyInstaller") is None:
        message = (
            "PyInstaller is not installed. Use start_control_room.bat or install PyInstaller "
            "in the presentation environment before packaging.\n"
        )
        print(message, end="")
        log_path.write_text(message, encoding="utf-8")
        return 2

    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(command, cwd=root, text=True, stdout=log_file, stderr=subprocess.STDOUT)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
