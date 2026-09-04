from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
NOTES_ROOT = REPO_ROOT / "experiments/notes"
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
STALE_STATUS_PATTERNS = (
    re.compile(r"306793[^\n]*was running"),
    re.compile(r"306796[^\n]*(?:was dependency-pending|was pending|dependency-pending)"),
    re.compile(r"306813/306814[^\n]*(?:submitted|had not run)"),
    re.compile(r"\*\*In flight\*\*"),
    re.compile(r"\.env is never read", re.IGNORECASE),
    re.compile(r"No script reads `?\.env", re.IGNORECASE),
    re.compile(r"Repository `\.env` files are never loaded", re.IGNORECASE),
)


def _markdown_files() -> tuple[Path, ...]:
    return tuple(sorted(NOTES_ROOT.rglob("*.md")))


def _relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def test_all_relative_markdown_links_resolve() -> None:
    missing: list[str] = []
    for note in _markdown_files():
        for line_number, line in enumerate(note.read_text(encoding="utf-8").splitlines(), start=1):
            for target in MARKDOWN_LINK.findall(line):
                path_text = target.partition("#")[0]
                if not path_text or "://" in path_text or path_text.startswith("mailto:"):
                    continue
                if not (note.parent / path_text).resolve().exists():
                    missing.append(f"{_relative(note)}:{line_number}: {target}")
    assert missing == []


def test_notes_readme_indexes_every_curated_note() -> None:
    readme = (NOTES_ROOT / "README.md").read_text(encoding="utf-8")
    linked_names = {
        target.partition("#")[0]
        for target in MARKDOWN_LINK.findall(readme)
        if "/" not in target and target.endswith(".md")
    }
    curated_names = {path.name for path in NOTES_ROOT.glob("*.md") if path.name != "README.md"}
    assert linked_names == curated_names


def test_curated_notes_do_not_restore_known_stale_status_claims() -> None:
    violations: list[str] = []
    for note in sorted(NOTES_ROOT.glob("*.md")):
        text = note.read_text(encoding="utf-8")
        for pattern in STALE_STATUS_PATTERNS:
            if match := pattern.search(text):
                line_number = text.count("\n", 0, match.start()) + 1
                violations.append(f"{_relative(note)}:{line_number}: {match.group(0)}")
    assert violations == []


def test_current_status_notes_name_the_recorded_validation_boundaries() -> None:
    domains = (NOTES_ROOT / "domains.md").read_text(encoding="utf-8")
    offline_eval = (NOTES_ROOT / "offline-eval.md").read_text(encoding="utf-8")

    assert "jobs 306793/306796 completed" in domains
    assert "Training uses only the pinned 1,982-row AReaL Tau2 RL split" in domains
    assert "Official Tau v3 train/base tasks are not" in domains
    assert "Workplace (single-turn multi-step)" in domains
    assert "conversational tool-use Pivot" in domains
    assert "Job 307365 completed the current unified runner" in offline_eval
    assert "Job 307366 generated" in offline_eval
    assert "Job 307427 completed" in offline_eval
