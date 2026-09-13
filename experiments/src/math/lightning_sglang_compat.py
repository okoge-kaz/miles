"""Apply the pinned OCI Nemotron routing-capture fix inside a writable container."""

import hashlib
import importlib.util
from pathlib import Path

SOURCE_SHA256 = "1882ff1678a714e8f5a1bca055f4e51a05b2c62611c5f4ac609900a6d7f9cd10"
BEFORE = "        self.topk = TopK(\n            top_k=config.num_experts_per_tok,\n"
AFTER = "        self.topk = TopK(\n            layer_id=layer_idx,\n            top_k=config.num_experts_per_tok,\n"


def patch_source(source: str) -> str:
    original = source.replace(AFTER, BEFORE)
    if hashlib.sha256(original.encode()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Unexpected SGLang Nemotron implementation; re-qualify the routing-capture patch")
    assert original.count(BEFORE) == 1
    return original.replace(BEFORE, AFTER)


def main():
    package = importlib.util.find_spec("sglang")
    path = Path(package.origin).parent / "srt/models/nemotron_h.py"
    source = path.read_text()
    patched = patch_source(source)
    if patched != source:
        path.write_text(patched)
    print(f"Nemotron routing-capture overlay: {path}; sha256={hashlib.sha256(patched.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
