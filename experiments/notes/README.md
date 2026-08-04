# notes/

Reference notes for running miles on cw-dfw. Two kinds of documents:

| Directory | Contents | Update policy |
|---|---|---|
| `notes/*.md` | Curated reference — how things work, how to inspect them | Edit in place; keep current |
| `notes/agents/*.md` | Work log — what was investigated/measured on a given day, with evidence | Append; never rewrite history |

Index:

- [cluster.md](cluster.md) — cw-dfw partitions, node specs, what is and is not installed
- [containers.md](containers.md) — docker image → `.sqsh`, pyxis flags, mount layout
- [datasets.md](datasets.md) — where the data lives, how to inspect it, how miles reads it
- [checkpoints.md](checkpoints.md) — `hf/` `megatron/` `training/`, conversion, resume
- [miles-architecture.md](miles-architecture.md) — the four objects, the directory map, the plug points

> **Caution.** This directory lives inside a checkout of an open-source repo. Keep
> internal endpoints, API keys, hostnames and verbatim internal discussion out of
> it — reference the channel or ticket instead.
