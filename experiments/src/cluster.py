"""Check release revisions, required GPU packages, visibility and BF16 matmul."""

import os
import subprocess


def _check_revision(path: str, expected: str) -> None:
    actual = subprocess.check_output(
        ["git", "-c", f"safe.directory={path}", "-C", path, "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != expected:
        raise RuntimeError(f"revision mismatch at {path}: expected {expected}, got {actual}")


def main() -> None:
    _check_revision("/sgl-workspace/sglang", os.environ["SGLANG_COMMIT"])
    _check_revision("/root/Megatron-LM", os.environ["MEGATRON_COMMIT"])

    # Load GPU libraries only after the release-pin checks have passed.
    import flash_attn  # noqa: F401 -- availability check
    import ray
    import sglang
    import torch
    import transformer_engine  # noqa: F401 -- availability check

    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == int(os.environ["GPUS_PER_NODE"])
    left = torch.randn((1024, 1024), device="cuda", dtype=torch.bfloat16)
    result = left @ left
    torch.cuda.synchronize()
    assert torch.isfinite(result).all()
    print(
        {
            "torch": torch.__version__,
            "sglang": sglang.__version__,
            "ray": ray.__version__,
            "gpu": torch.cuda.get_device_name(0),
        }
    )


if __name__ == "__main__":
    main()
