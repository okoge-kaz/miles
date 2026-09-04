# Experiment implementation layout

- `environments/`: rollout runtimes, generators, sandboxes, and verifiers;
- `datasets/`: source adapters, conversion, audit, merge, and preparation CLIs;
- `reward_sets/`: fail-closed recipe entry points and shared static rewards;
- `protocols/`: external request/message schema translation;
- `evaluators/`: held-out offline benchmark evaluation adapters.

Only these canonical namespaces are supported. Dataset preparation and
environment verification are imported directly from their owning packages;
there are no forwarding aliases between them.
