"""Block until `want` Ray nodes are alive, or fail.

    python3 wait_for_ray_nodes.py <address> <want> [timeout_seconds]

miles sizes its placement groups from the node count, so a job submitted before
every worker has registered silently runs smaller than its allocation. Reachable
is not the same as registered, which is why this checks ray.nodes() rather than
the head's port.
"""

import sys
import time

import ray


def main() -> int:
    address = sys.argv[1]
    want = int(sys.argv[2])
    timeout = float(sys.argv[3]) if len(sys.argv) > 3 else 600.0

    ray.init(address=address)
    deadline = time.time() + timeout
    while True:
        alive = sum(1 for node in ray.nodes() if node["Alive"])
        if alive >= want:
            print(f"ray cluster ready: {alive}/{want} nodes", flush=True)
            return 0
        if time.time() > deadline:
            print(f"only {alive}/{want} ray nodes joined within {timeout:.0f}s", file=sys.stderr)
            return 1
        print(f"waiting for ray nodes: {alive}/{want}", flush=True)
        time.sleep(5)


if __name__ == "__main__":
    sys.exit(main())
