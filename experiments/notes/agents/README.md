# notes/agents/ — work log

Append-only record of what was actually investigated, measured or decided, split
by area. The curated "how it works" version of the same material lives one level
up in `notes/`; this directory keeps the evidence and the dates so a claim can be
re-checked later.

Convention: newest entry first inside each file, `## YYYY-MM-DD — topic`, and
every factual claim carries the command that produced it or a file:line
reference. If something was **not** verified, say so explicitly.

| File | Area |
|---|---|
| [cluster.md](cluster.md) | Slurm, partitions, container runtime, network |
| [miles-implementation.md](miles-implementation.md) | Reading of the miles codebase |
| [dataset.md](dataset.md) | Data sources, formats, how miles ingests them |
| [checkpoints.md](checkpoints.md) | Formats, conversion, resume behaviour |
| [container.md](container.md) | Image import, pyxis/enroot mechanics |
| [sandbox-and-agentic-rl.md](sandbox-and-agentic-rl.md) | Sandbox options for agentic RL |

> Keep internal endpoints, keys and verbatim internal discussion out of these
> files — this is a checkout of an open-source repo.
