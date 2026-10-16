import argparse
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class FilterConfig:
    model_path: str
    prompt_data: str
    measurement_dir: str
    filtered_output: str
    policy: str
    reward_backend: str = "deepscaler"
    selection_rule: str = "mixed_correctness"
    n_samples: int = 16
    max_new_tokens: int = 16384
    max_context_length: int = 32768
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    seed: int = 1234
    correct_threshold: float = 0.5
    zero_reward_on_truncated: bool = True
    concurrency: int = 160
    tensor_parallel_size: int = 1
    num_engines: int = 8
    max_running_requests: int = 80
    mem_fraction_static: float = 0.85
    num_shards: int = 4
    server_port: int = 30000
    startup_timeout: int = 900
    request_timeout: int = 3600
    audit_prompts: int = 4
    generation_attempts: int = 5
    progress_every: int = 10
    progress_seconds: int = 60
    gym_port: int = 30190
    code_concurrency: int = 16

    def __post_init__(self):
        for name in (
            "n_samples",
            "max_new_tokens",
            "max_context_length",
            "concurrency",
            "tensor_parallel_size",
            "num_engines",
            "max_running_requests",
            "num_shards",
            "startup_timeout",
            "request_timeout",
            "generation_attempts",
            "progress_every",
            "progress_seconds",
            "code_concurrency",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_samples < 2:
            raise ValueError("Require n_samples >= 2")
        if self.max_new_tokens > self.max_context_length:
            raise ValueError("max_new_tokens must not exceed max_context_length")
        for name in ("temperature", "top_p", "correct_threshold", "mem_fraction_static"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.top_p > 1 or self.correct_threshold > 1 or self.mem_fraction_static >= 1:
            raise ValueError("Require top_p/threshold <= 1 and mem_fraction_static < 1")
        if self.top_k != -1 and self.top_k <= 0:
            raise ValueError("top_k must be -1 or positive")
        if self.seed < 0 or self.audit_prompts < 0 or not 1 <= self.server_port <= 65535:
            raise ValueError("Invalid seed, audit_prompts, or server_port")
        engine_ports = {self.server_port + 10 * index for index in range(self.num_engines)}
        if max(engine_ports) > 65535 or engine_ports & {self.gym_port, self.gym_port + 1}:
            raise ValueError("Engine ports must be valid and separate from Gym ports")
        if type(self.zero_reward_on_truncated) is not bool:
            raise ValueError("zero_reward_on_truncated must be boolean")
        for name in ("model_path", "prompt_data", "measurement_dir", "filtered_output"):
            if not Path(getattr(self, name)).is_absolute():
                raise ValueError(f"{name} must be an absolute path")
        if self.prompt_data == self.filtered_output:
            raise ValueError("The filtered output must differ from the source")
        if not self.policy:
            raise ValueError("policy must not be empty")
        if self.reward_backend not in {"deepscaler", "code_gen", "mcqa", "instruction_following"}:
            raise ValueError("Unknown reward backend")
        if self.selection_rule not in {"mixed_correctness", "reward_variance"}:
            raise ValueError("Unknown selection rule")
        if self.reward_backend == "instruction_following" and (
            self.selection_rule != "reward_variance" or self.correct_threshold != 1
        ):
            raise ValueError("Instruction following requires reward_variance and full-success threshold 1")
        if not 1 <= self.gym_port <= 65534 or self.server_port in {self.gym_port, self.gym_port + 1}:
            raise ValueError("Gym ports must be valid and separate from SGLang")

    def measurement_settings(self):
        # Output location and operational timeouts do not change the measured distribution.
        ignored = {
            "measurement_dir",
            "filtered_output",
            "server_port",
            "startup_timeout",
            "request_timeout",
            "generation_attempts",
            "progress_every",
            "progress_seconds",
            "gym_port",
        }
        return {key: value for key, value in asdict(self).items() if key not in ignored}


def load_config(path):
    return FilterConfig(**json.loads(Path(path).read_text()))


def main():
    parser = argparse.ArgumentParser(description="Validate and serialize an offline difficulty-filter recipe")
    required = {"model_path", "prompt_data", "measurement_dir", "filtered_output", "policy"}
    for field in fields(FilterConfig):
        name = "--" + field.name.replace("_", "-")
        if field.type is bool:
            parser.add_argument(name, action=argparse.BooleanOptionalAction, default=field.default)
        elif field.name in required:
            parser.add_argument(name, required=True)
        else:
            parser.add_argument(name, type=field.type, default=field.default)
    config = FilterConfig(**vars(parser.parse_args()))
    print(json.dumps(asdict(config), indent=2))


if __name__ == "__main__":
    main()
