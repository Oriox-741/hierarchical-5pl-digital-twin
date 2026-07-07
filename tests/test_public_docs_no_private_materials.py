from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

BLOCKED_PATH_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"transcript",
        r"transkrpit",
        r"Academic_CV",
        r"Research_Summary_GRA",
        r"Working_Paper_Version",
        r"Ablation_Addendum",
        r"GRA_Outreach_Pack",
        r"Implementation_Playbook",
        r"\.docx$",
    )
]

TEXT_SUFFIXES = {".md", ".txt", ".rst", ".csv", ".json", ".yml", ".yaml", ".toml", ".cfg", ".ini"}
BLOCKED_TEXT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bstudent\s*id\b",
        r"\btc\s*kimlik\b",
        r"og[rğ]enci\s*numaras[ıi]",
        r"\bAGNO\b",
        r"\bGPA\b",
        r"\bphone\s*:",
        r"\btelephone\s*:",
        r"\b(home|personal|residential)\s+address\s*:",
    )
]


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line]


class PublicDocsPrivacyTests(unittest.TestCase):
    def test_tracked_files_do_not_include_private_application_materials(self) -> None:
        tracked = _tracked_files()
        violations = [
            path for path in tracked if any(pattern.search(path.replace("\\", "/")) for pattern in BLOCKED_PATH_PATTERNS)
        ]
        self.assertEqual([], violations, f"Tracked private application or DOCX files found: {violations}")

    def test_tracked_text_files_do_not_expose_private_identity_labels(self) -> None:
        violations: list[str] = []
        for rel_path in _tracked_files():
            path = REPO_ROOT / rel_path
            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for pattern in BLOCKED_TEXT_PATTERNS:
                if pattern.search(text):
                    violations.append(f"{rel_path}: {pattern.pattern}")
        self.assertEqual([], violations, "Tracked text files include private identity labels: " + ", ".join(violations))


if __name__ == "__main__":
    unittest.main()
