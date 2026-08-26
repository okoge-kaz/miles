"""Live semantic admission for immutable SWE-ReBench-V2 E2B tasks."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiments.src.environments.swe import e2b_admission
from experiments.src.environments.swe import materialize as materialize_module
from experiments.src.environments.swe import semantic_admission

ADMISSION_SCHEMA = "miles-swe-rebench-admission-v1"
_CHECKS: dict[str, bool | int] = {
    "publisher_namespace_policy": True,
    "registry_digest_resolved": True,
    "agent_verifier_same_image_digest": True,
    "source_head_matches_base": True,
    "agent_history_scrubbed": True,
    "verifier_history_scrubbed": True,
    "hidden_test_patch_isolated": True,
    "late_private_verifier_upload": True,
    "model_path_policy_enforced": True,
    "safe_patch_policy_enforced": True,
    "official_test_command_parity": True,
    "official_exact_parser_parity": True,
    "empty_reward": 0,
    "oracle_reward": 1,
    "runtime_smoke": True,
    "tool_smoke": True,
    "no_new_privileges": True,
    "effective_capabilities_zero": True,
    "fresh_separate_verifier": True,
    "fresh_template_id_pinned": True,
    "agent_public_network_blocked": True,
    "verifier_public_network_blocked": True,
    "tracked_gold_history_absent": True,
}


@dataclass(frozen=True)
class RebenchAdapter:
    """Pinned SWE-ReBench parser, evaluator, and task-tree contract."""

    log_parsers: Path
    constants: Path
    official_eval: Path
    source_schema: str = "swe-rebench-v2"
    admission_schema: str = ADMISSION_SCHEMA
    checkpoint_label: str = "rebench-admit"
    report_kind: str = "swe-rebench-v2"
    required_checks: Mapping[str, bool | int] = field(
        default_factory=lambda: dict(_CHECKS)
    )

    def validate_dependencies(self) -> None:
        _require_dependency(
            self.log_parsers,
            materialize_module._REBENCH_LOG_PARSERS_SHA256,
            "SWE-ReBench log parser",
        )
        _require_dependency(
            self.constants,
            materialize_module._REBENCH_CONSTANTS_SHA256,
            "SWE-ReBench constants",
        )
        _require_dependency(
            self.official_eval,
            materialize_module._REBENCH_EVAL_SHA256,
            "SWE-ReBench evaluator",
        )

    def validate_candidate(self, row: Mapping[str, Any]) -> None:
        verifier = _mapping(row, "verifier")
        if verifier.get("kind") != "swe-rebench-v2":
            raise semantic_admission.QuarantineTask(
                "unsupported_verifier_kind",
                "SWE-ReBench verifier kind is unsupported",
            )
        install_config = _mapping(verifier, "install_config")
        commands = _test_commands(install_config.get("test_cmd"))
        parser = _text(install_config, "log_parser")
        if re.fullmatch(r"parse_[A-Za-z0-9_]+", parser) is None:
            raise semantic_admission.QuarantineTask(
                "unsupported_log_parser",
                "SWE-ReBench parser name is not explicit",
            )
        if not commands:
            raise semantic_admission.QuarantineTask(
                "test_command_missing",
                "SWE-ReBench test command is absent",
            )
        _string_list(verifier.get("fail_to_pass"), "fail_to_pass")
        _string_list(verifier.get("pass_to_pass"), "pass_to_pass")

    def materialize_arguments(
        self,
        manifest: Path,
        output: Path,
    ) -> argparse.Namespace:
        return argparse.Namespace(
            manifest=manifest,
            output=output,
            admission_evidence=None,
            r2e_execution_log_parser=None,
            r2e_admission_manifest=None,
            swe_rebench_log_parsers=self.log_parsers,
            swe_rebench_constants=self.constants,
            swe_rebench_eval=self.official_eval,
            swe_rebench_admission_manifest=None,
            swe_gym_harness_root=None,
            swe_gym_admission_manifest=None,
            allow_mutable_images=False,
            allow_unadmitted_r2e_dry_run=False,
            allow_unadmitted_swe_rebench_dry_run=True,
            allow_unadmitted_swe_gym_dry_run=False,
            limit=None,
            summary=None,
        )

    def validate_materialized(
        self,
        task_dir: Path,
        row: Mapping[str, Any],
    ) -> None:
        verifier = _mapping(row, "verifier")
        install_config = _mapping(verifier, "install_config")
        config = _read_json(task_dir / "tests" / "verifier_config.json")
        expected = {
            "instance_id": _text(row, "instance_id"),
            "base_commit": _text(row, "base_commit"),
            "test_commands": _test_commands(install_config.get("test_cmd")),
            "log_parser": _text(install_config, "log_parser"),
            "fail_to_pass": _string_list(
                verifier.get("fail_to_pass"),
                "fail_to_pass",
            ),
            "pass_to_pass": _string_list(
                verifier.get("pass_to_pass"),
                "pass_to_pass",
            ),
            "parser_commit": materialize_module._REBENCH_COMMIT,
            "log_parsers_sha256": materialize_module._REBENCH_LOG_PARSERS_SHA256,
            "constants_sha256": materialize_module._REBENCH_CONSTANTS_SHA256,
            "eval_sha256": materialize_module._REBENCH_EVAL_SHA256,
        }
        if config != expected:
            raise semantic_admission.QuarantineTask(
                "official_verifier_binding_mismatch",
                "materialized SWE-ReBench verifier differs from pinned inputs",
            )
        test_patch = _text(verifier, "test_patch")
        if (task_dir / "tests" / "test_patch.diff").read_text(
            encoding="utf-8"
        ) != test_patch:
            raise semantic_admission.QuarantineTask(
                "private_test_binding_mismatch",
                "materialized hidden patch differs from the private record",
            )

    def validate_report(
        self,
        row: Mapping[str, Any],
        report: Mapping[str, Any],
        reward: int,
    ) -> None:
        if reward == 0 and "kind" not in report:
            return
        parser = _text(_mapping(_mapping(row, "verifier"), "install_config"), "log_parser")
        if report.get("kind") != self.report_kind or report.get("parser") != parser:
            raise semantic_admission.SystemicAdmissionError(
                "SWE-ReBench report did not use the pinned official parser"
            )

    def admission_metadata(self, row: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "rebench_commit": materialize_module._REBENCH_COMMIT,
            "log_parsers_sha256": materialize_module._REBENCH_LOG_PARSERS_SHA256,
            "constants_sha256": materialize_module._REBENCH_CONSTANTS_SHA256,
            "eval_sha256": materialize_module._REBENCH_EVAL_SHA256,
        }


def _require_dependency(path: Path, expected: str, name: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"pinned {name} is missing: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"pinned {name} checksum mismatch")


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, dict):
        raise semantic_admission.QuarantineTask(
            "required_metadata_missing",
            f"required object {key} is absent",
        )
    return result


def _text(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise semantic_admission.QuarantineTask(
            "required_metadata_missing",
            f"required text {key} is absent",
        )
    return result.strip()


def _string_list(value: Any, key: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise semantic_admission.QuarantineTask(
            "invalid_test_inventory",
            f"{key} is not a string list",
        )
    return value


def _test_commands(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or any(
        not isinstance(command, str) or not command.strip() for command in value
    ):
        raise semantic_admission.QuarantineTask(
            "invalid_test_command",
            "install_config.test_cmd is invalid",
        )
    return [command for command in value]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise semantic_admission.QuarantineTask(
            "official_verifier_binding_mismatch",
            "materialized verifier config is not an object",
        )
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-manifest", type=Path, required=True)
    parser.add_argument("--admitted-manifest", type=Path, required=True)
    parser.add_argument("--admission-manifest", type=Path, required=True)
    parser.add_argument("--quarantine-manifest", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--swe-rebench-log-parsers", type=Path, required=True)
    parser.add_argument("--swe-rebench-constants", type=Path, required=True)
    parser.add_argument("--swe-rebench-eval", type=Path, required=True)
    parser.add_argument("--instance-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    return args


def main() -> None:
    args = parse_args()
    config = semantic_admission.AdmissionConfig(
        private_manifest=args.private_manifest,
        admitted_manifest=args.admitted_manifest,
        admission_manifest=args.admission_manifest,
        quarantine_manifest=args.quarantine_manifest,
        work_root=args.work_root,
        instance_id=args.instance_id,
        limit=args.limit,
        concurrency=args.concurrency,
    )
    adapter = RebenchAdapter(
        log_parsers=args.swe_rebench_log_parsers,
        constants=args.swe_rebench_constants,
        official_eval=args.swe_rebench_eval,
    )
    summary = asyncio.run(
        semantic_admission.admit_tasks(
            config,
            adapter,
            e2b_admission.NativeHarborE2BBackend(),
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
