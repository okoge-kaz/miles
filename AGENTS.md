# Codex Instructions

Before working in this repository, discover and read all Markdown rule files under `.agents/rules/`, including subdirectories. This repository-root-relative path is a symlink to `.claude/rules/`, the shared source of truth. Treat these files as repository instructions; newly added rules belong to the same set.

Apply each rule according to its `paths` frontmatter, matching globs against repository-relative file paths. Rules without `paths` apply repository-wide. Preserve any further applicability conditions in the rule body, such as applying only to new or substantially modified code.

Recheck which rules apply when the task expands to additional files or components.

## Code Review Rules

For pull request reviews, also read and follow `REVIEW.md` at the repository root.

Per user preference, do not add opening module docstrings to new Python files. Put design explanations in the relevant notes and retain useful function/class documentation.

For ABCI experiment implementation, debugging, or notes, use
`.claude/skills/abci-experiments/SKILL.md` and consult the relevant files in
`experiments/notes/agents/`. Keep analysis and measurements in `experiments/notes/logs/`.
