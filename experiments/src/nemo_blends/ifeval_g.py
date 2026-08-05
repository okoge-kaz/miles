"""Instruction-following verifier backed by open-instruct's IFEvalG registry.

miles ships `--rm-type ifbench`, which loads allenai/IFBench's registry. That is
the wrong registry for `nvidia/Nemotron-RL-instruction_following`: the two share
**no instruction ids at all**. Measured against the staged 20k subset:

    IFBench registry        58 ids   count:conjunctions, custom:csv_city,
                                     format:emoji, ratio:overlap, repeat:repeat_span
    dataset                 48 ids   keywords:existence, punctuation:no_comma,
                                     detectable_format:title, change_case:english_lowercase
    overlap                  0       (1 of 48 even if the family prefix is ignored)
    rows fully gradable      0/20096

IFBench introduced a new constraint family; the dataset uses the IFEval family.
The dataset card says as much -- it pairs WildChat prompts with "instructions
from the Open-Instruct code base" -- and open-instruct's IFEvalG registry covers
**48 of 48**, where google-research's original IFEval covers only 24.

So this is not a data problem. The `instruction_id_list` / `prompt_text` /
`kwargs` metadata the converter already writes is exactly right; what was
missing is the code that implements those 48 checkers.

    --custom-rm-path experiments.src.nemo_blends.ifeval_g.ifeval_reward

Training then grades against IFEval-family constraints while `--rm-type ifbench`
still works as an *eval* (verified: 300/300 of allenai/IFBench_test is gradable).
That split is a feature -- IFBench is the harder, out-of-distribution constraint
set the paper intends it to be -- but it has to be stated when reporting: the
train and eval constraint families are different.
"""

import logging
import os
import subprocess
import sys
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["ifeval_reward", "build_preflight_probes"]

_REPO_URL = "https://github.com/allenai/open-instruct.git"
# The container's /root is ephemeral, so this re-clones per job. Kept shallow and
# single-branch: the checkout is ~50 MB and only one module is imported from it.
_REPO_PATH = Path(os.environ.get("OPEN_INSTRUCT_PATH", "/root/open-instruct"))
_MODULE = "open_instruct.IFEvalG.instructions_registry"

_lock = threading.Lock()
_registry = None


def _load_registry():
    """Import IFEvalG's registry, cloning open-instruct if it is not there yet.

    Mirrors what `miles/rollout/rm_hub/ifbench.py` does for IFBench, including
    the reason: the registry is not on PyPI in any form, and vendoring 48
    checkers into this repo would fork them from upstream silently.
    """
    global _registry
    if _registry is not None:
        return _registry
    with _lock:
        if _registry is not None:
            return _registry

        if not _REPO_PATH.exists():
            try:
                subprocess.run(
                    ["git", "clone", "--depth", "1", "--filter=blob:none", _REPO_URL, str(_REPO_PATH)],
                    check=True,
                    capture_output=True,
                )
            except Exception as exc:
                raise ImportError(
                    f"could not clone open-instruct into {_REPO_PATH}. Clone it there yourself, "
                    f"or set OPEN_INSTRUCT_PATH to an existing checkout."
                ) from exc

        repo = str(_REPO_PATH)
        if repo not in sys.path:
            sys.path.insert(0, repo)

        import importlib

        _registry = importlib.import_module(_MODULE)
        logger.info("IFEvalG registry: %d instruction ids", len(_registry.INSTRUCTION_DICT))
        return _registry


def _score_one(sample) -> float:
    """1.0 only when *every* constraint on the prompt holds.

    IFEval's own strict metric, and the right one here: the constraints on a
    prompt are conjunctive, so partial credit would reward a policy that learns
    the easy half of each prompt and ignores the rest.
    """
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    instruction_ids = metadata.get("instruction_id_list") or []
    if not instruction_ids:
        return 0.0
    response = sample.response or ""
    # An empty response satisfies some constraints vacuously ("contains no
    # comma"), so it has to be rejected before the checkers run.
    if not response.strip():
        return 0.0

    registry = _load_registry()
    kwargs_list = metadata.get("kwargs") or []
    prompt_text = str(metadata.get("prompt_text") or "")

    for index, instruction_id in enumerate(instruction_ids):
        cls = registry.INSTRUCTION_DICT.get(instruction_id)
        if cls is None:
            # A prompt we cannot fully grade must not be scored as satisfied:
            # that would hand out free reward for the ungradable rows.
            logger.warning("unknown instruction id %s; scoring 0", instruction_id)
            return 0.0
        checker = cls(instruction_id)
        kwargs = kwargs_list[index] if index < len(kwargs_list) else {}
        checker.build_description(**{k: v for k, v in (kwargs or {}).items() if v is not None})
        # Some checkers quote the prompt back (e.g. "repeat the request first").
        if hasattr(checker, "get_instruction_args_keys") and "prompt" in (checker.get_instruction_args_keys() or []):
            checker.build_description(prompt=prompt_text)
        try:
            if not checker.check_following(response):
                return 0.0
        except Exception as exc:  # noqa: BLE001 - a checker raising is a bad row, not a policy failure
            logger.warning("instruction %s raised %s; scoring 0", instruction_id, type(exc).__name__)
            return 0.0
    return 1.0


async def ifeval_reward(args, sample_or_samples, **kwargs):
    """Accepts both miles custom-rm contracts (a list from `batched_async_rm`,
    one Sample from `async_rm`)."""
    if isinstance(sample_or_samples, list):
        return [_score_one(s) for s in sample_or_samples]
    return _score_one(sample_or_samples)


def build_preflight_probes(label, metadata):
    """No synthesizable correct answer exists here.

    The constraints are arbitrary and per-row ("four paragraphs, the third
    starting with 'crash'"), so any generic probe fails and a strict preflight
    would refuse every sweep. Returning None tells the driver to say so rather
    than to treat it as a verifier fault; run these with `--preflight warn`.
    """
    return None
