# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/ceee7b89655ed52f205b9beb98e1190c3eedcfb0/search_r1/llm_agent/generation.py
# This is a unified version supporting both local search and Google search, with optional log probability collection

import asyncio
import logging
import os
import re

from qa_em_format import compute_score_em

from miles.rollout.sglang_rollout import GenerateState
from miles.utils.http_utils import post
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

REQUIRED_STOP_STRINGS = frozenset({"</search>", "</answer>"})


class SearchBackendError(RuntimeError):
    """A search action could not obtain a valid observation."""


# Configuration for Search-R1
SEARCH_R1_CONFIGS = {
    # ============== General Configuration ==============
    "max_turns": 2,
    "topk": 3,
    "search_concurrency": 256,
    "search_timeout": 60,
    "search_max_attempts": 3,
    # ============== Search Backend Selection ==============
    "search_backend": "local",  # Options: "local" or "google"
    # ============== Local Search Configuration ==============
    # (Only used when search_backend="local")
    "local": {
        "search_url": "http://127.0.0.1:8000/retrieve",  # URL of your local retrieval server
        "proxy": None,  # Set to your proxy if needed
    },
    # ============== Google Search Configuration ==============
    # (Only used when search_backend="google")
    "google": {
        "api_key": "your_api_key_here",  # Replace with your actual API key
        "snippet_only": True,  # Set to True to only return snippets
        "proxy": None,  # Set to your proxy if needed
    },
    # ============== Log Probability Collection ==============
    "return_logprob": True,  # Set to True to collect log probabilities for TIS metrics
    # ============== Reward Model Configuration ==============
    "format_score": 0.2,
}


# Environment overrides, applied at import.
#
# Without these the dict above is the only way to configure a run, which means a
# recipe cannot say where its retriever is: the URL stays 127.0.0.1 no matter what
# the job exports. That is not a loud failure -- `search()` turns an unreachable
# retriever into an empty observation, so the rollout completes, rewards come back
# non-zero from the model's own knowledge, and nothing in the metrics says the
# retriever was never used.
def _env_int(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from error
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _env_float(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a number, got {raw!r}") from error
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")
    return value


SEARCH_R1_CONFIGS["max_turns"] = _env_int("SEARCH_R1_MAX_TURNS", SEARCH_R1_CONFIGS["max_turns"])
SEARCH_R1_CONFIGS["topk"] = _env_int("SEARCH_R1_TOPK", SEARCH_R1_CONFIGS["topk"])
SEARCH_R1_CONFIGS["search_concurrency"] = _env_int(
    "SEARCH_R1_SEARCH_CONCURRENCY", SEARCH_R1_CONFIGS["search_concurrency"]
)
SEARCH_R1_CONFIGS["search_timeout"] = _env_int("SEARCH_R1_SEARCH_TIMEOUT", SEARCH_R1_CONFIGS["search_timeout"])
SEARCH_R1_CONFIGS["search_max_attempts"] = _env_int(
    "SEARCH_R1_SEARCH_MAX_ATTEMPTS", SEARCH_R1_CONFIGS["search_max_attempts"]
)
SEARCH_R1_CONFIGS["format_score"] = _env_float("SEARCH_R1_FORMAT_SCORE", SEARCH_R1_CONFIGS["format_score"])
if os.environ.get("SEARCH_R1_SEARCH_URL"):
    SEARCH_R1_CONFIGS["local"]["search_url"] = os.environ["SEARCH_R1_SEARCH_URL"]

SEMAPHORE = asyncio.Semaphore(SEARCH_R1_CONFIGS["search_concurrency"])


def _passages2string(retrieval_result):
    """
    Convert retrieval results to a formatted string.
    This function works with both google_search and local_search results.
    """
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        try:
            content = doc_item["document"]["contents"]
        except (KeyError, TypeError) as error:
            raise SearchBackendError(f"search result {idx} has no document contents") from error
        if not isinstance(content, str) or not content.strip():
            raise SearchBackendError(f"search result {idx} has empty document contents")
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx + 1}(Title: {title}) {text}\n"

    return format_reference


async def search(query: str) -> str:
    """
    Perform search using either local search engine or Google search.
    The search backend is determined by SEARCH_R1_CONFIGS["search_backend"].
    """
    backend = SEARCH_R1_CONFIGS["search_backend"]

    try:
        if backend == "local":
            from local_search_server import local_search

            local_config = SEARCH_R1_CONFIGS["local"]
            result = await local_search(
                local_config["search_url"],
                query,
                SEARCH_R1_CONFIGS["topk"],
                timeout=SEARCH_R1_CONFIGS["search_timeout"],
                proxy=local_config["proxy"],
                max_attempts=SEARCH_R1_CONFIGS["search_max_attempts"],
            )
            if not result:
                raise SearchBackendError("local retriever returned no passages")
        elif backend == "google":
            from google_search_server import google_search

            google_config = SEARCH_R1_CONFIGS["google"]
            result = await google_search(
                google_config["api_key"],
                query,
                SEARCH_R1_CONFIGS["topk"],
                timeout=SEARCH_R1_CONFIGS["search_timeout"],
                snippet_only=google_config["snippet_only"],
                proxy=google_config["proxy"],
            )
        else:
            raise ValueError(f"Unknown search backend: {backend}. Must be either 'local' or 'google'.")
    except SearchBackendError:
        raise
    except Exception as error:
        raise SearchBackendError(f"{backend} search failed: {error}") from error

    return _passages2string(result)


# IMPORTANT: When we need to collect log probabilities (logp), we CANNOT do any postprocessing
# on the strings returned from the inference engine (sglang). This is because:
# 1. We don't know how to truncate the corresponding tokens/logp arrays to match the modified string
# 2. Re-tokenizing the postprocessed string may produce different tokens than what the engine generated,
#    leading to misalignment between tokens and their log probabilities
# Therefore, postprocess_responses is only used when return_logprob=False.
def postprocess_responses(resp: str) -> str:
    """
    Post-process response to ensure tag completeness.
    Only used when SEARCH_R1_CONFIGS["return_logprob"] is False.
    """
    return (
        resp.split("</search>")[0] + "</search>"
        if "</search>" in resp
        else resp.split("</answer>")[0] + "</answer>"
        if "</answer>" in resp
        else resp
    )


def postprocess_predictions(prediction: str):
    pattern = r"<(search|answer)>(.*?)</\1>"
    match = re.search(pattern, prediction, re.DOTALL)
    if match:
        content = match.group(2).strip()  # Return only the content inside the tags
        action = match.group(1)
    else:
        content = ""
        action = None

    return action, content


async def execute_predictions(prediction: str) -> tuple[str, bool]:
    action, content = postprocess_predictions(prediction)

    if action == "search" and content:
        search_query = content
        async with SEMAPHORE:
            search_results = await search(search_query)
        next_obs = f"\n\n<information>{search_results.strip()}</information>\n\n"
        done = False
    elif action == "answer":
        next_obs = ""
        done = True
    else:
        next_obs = "\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n"
        done = False

    return next_obs, done


def _validate_sampling_params(sampling_params: dict) -> None:
    stops = sampling_params.get("stop") or []
    if isinstance(stops, str):
        stops = [stops]
    missing = REQUIRED_STOP_STRINGS.difference(stops)
    if missing:
        raise ValueError(
            f"Search-R1 requires SGLang to stop at both action boundaries; missing stop strings: {sorted(missing)}"
        )
    if sampling_params.get("no_stop_trim") is not True:
        raise ValueError(
            "Search-R1 requires no_stop_trim=True so returned text and token/logprob arrays retain the tag"
        )


def _populate_sample(
    sample: Sample,
    prompt_text: str,
    prompt_token_ids: list[int],
    response: str,
    response_token_ids: list[int],
    loss_mask: list[int],
    rollout_log_probs: list[float] | None,
) -> None:
    sample.tokens = prompt_token_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_mask
    sample.prompt = prompt_text
    sample.rollout_log_probs = rollout_log_probs
    sample.validate()


def _record_rollout_metrics(sample: Sample, *, turns: int, search_calls: int, error: str | None = None) -> None:
    sample.metadata = dict(sample.metadata or {})
    sample.metadata["search_r1_turns"] = turns
    sample.metadata["search_r1_search_calls"] = search_calls
    if error is not None:
        sample.metadata["search_r1_error"] = error


async def generate(args, sample: Sample, sampling_params) -> Sample:
    assert not args.partial_rollout, "Partial rollout is not supported for this function at the moment."
    _validate_sampling_params(sampling_params)

    state = GenerateState(args)

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    prompt_text = sample.prompt
    if not isinstance(prompt_text, str):
        raise TypeError(
            "Search-R1 requires a rendered string prompt; pass --apply-chat-template for the training parquet"
        )
    prompt_token_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    response = ""
    response_token_ids = []
    loss_mask = []
    rollout_log_probs = [] if SEARCH_R1_CONFIGS["return_logprob"] else None
    turns_used = 0
    search_calls = 0

    for turn_index in range(SEARCH_R1_CONFIGS["max_turns"]):
        payload = {
            "text": prompt_text + response,
            "sampling_params": sampling_params,
        }
        # Add log probability collection if enabled
        if SEARCH_R1_CONFIGS["return_logprob"]:
            payload["return_logprob"] = True

        output = await post(url, payload)
        meta_info = output["meta_info"]
        finish_type = meta_info["finish_reason"]["type"]
        turns_used = turn_index + 1
        sample.update_from_meta_info(args, meta_info)

        # abort
        if finish_type == "abort":
            _populate_sample(
                sample,
                prompt_text,
                prompt_token_ids,
                response,
                response_token_ids,
                loss_mask,
                rollout_log_probs,
            )
            _record_rollout_metrics(sample, turns=turns_used, search_calls=search_calls, error="sglang_abort")
            return sample

        cur_response = output["text"]

        # Extract tokens and log probs based on configuration
        if SEARCH_R1_CONFIGS["return_logprob"]:
            # Extract log probs from output - required for TIS metrics
            if "output_token_logprobs" not in meta_info:
                raise RuntimeError(
                    "output_token_logprobs not found in output meta_info. "
                    "Make sure 'return_logprob': True is set in the payload."
                )

            # Use token IDs and log probs directly from output_token_logprobs
            # This ensures perfect alignment between tokens and log probs
            # output_token_logprobs format: [[log_prob, token_id, ...], ...]
            cur_response_token_ids = [item[1] for item in meta_info["output_token_logprobs"]]
            cur_response_log_probs = [item[0] for item in meta_info["output_token_logprobs"]]
        else:
            # When not collecting log probs, we can safely postprocess the response
            cur_response = postprocess_responses(cur_response)
            # Tokenize the (possibly postprocessed) response
            cur_response_token_ids = state.tokenizer(cur_response, add_special_tokens=False)["input_ids"]

        response += cur_response
        response_token_ids += cur_response_token_ids
        loss_mask += [1] * len(cur_response_token_ids)

        # Add log probs if enabled
        if SEARCH_R1_CONFIGS["return_logprob"]:
            rollout_log_probs += cur_response_log_probs
            assert len(response_token_ids) == len(rollout_log_probs), (
                f"Token/logp length mismatch: {len(response_token_ids)} tokens vs {len(rollout_log_probs)} logps"
            )

        if finish_type == "length":
            break

        action, _ = postprocess_predictions(cur_response)
        if action == "search":
            search_calls += 1
        try:
            next_obs, done = await execute_predictions(cur_response)
        except SearchBackendError as error:
            sample.status = Sample.Status.ABORTED
            _populate_sample(
                sample,
                prompt_text,
                prompt_token_ids,
                response,
                response_token_ids,
                loss_mask,
                rollout_log_probs,
            )
            _record_rollout_metrics(sample, turns=turns_used, search_calls=search_calls, error=str(error))
            logger.warning("Aborting Search-R1 trajectory after retriever failure: %s", error)
            return sample
        if done:
            break

        assert next_obs != "", "Next observation should not be empty."
        obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
        response += next_obs
        response_token_ids += obs_tokens_ids
        loss_mask += [0] * len(obs_tokens_ids)

        # Add dummy log probs for observation tokens if enabled (they won't be used due to loss_mask=0)
        if SEARCH_R1_CONFIGS["return_logprob"]:
            rollout_log_probs += [0.0] * len(obs_tokens_ids)

            # Verify alignment when collecting log probs
            assert len(response_token_ids) == len(rollout_log_probs), (
                f"Token/logp length mismatch: {len(response_token_ids)} tokens vs {len(rollout_log_probs)} logps"
            )

    _populate_sample(
        sample,
        prompt_text,
        prompt_token_ids,
        response,
        response_token_ids,
        loss_mask,
        rollout_log_probs,
    )
    _record_rollout_metrics(sample, turns=turns_used, search_calls=search_calls)

    return sample


async def reward_func(args, sample, **kwargs):
    """The reward function for retrieval-based question answering.

    Args:
        args: the arguments
        sample: the sample to evaluate
    """
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    score = compute_score_em(
        solution_str=sample.prompt + sample.response,
        ground_truth=sample.label["ground_truth"],
        structure_format_score=SEARCH_R1_CONFIGS["format_score"],
    )

    return score
