"""Live semantic admission for SWE-Gym and Nemotron SWE-Gym E2B tasks."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiments.src.environments.swe import e2b_admission
from experiments.src.environments.swe import materialize as materialize_module
from experiments.src.environments.swe import semantic_admission

ADMISSION_SCHEMA = "miles-swe-gym-admission-v1"
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
class SweGymAdapter:
    """Pinned SWE-Gym package, parser, and task-tree contract."""

    harness_root: Path
    source_schema: str = "swe-gym"
    admission_schema: str = ADMISSION_SCHEMA
    checkpoint_label: str = "swe-gym-admit"
    report_kind: str = "swe-gym-pinned-v2.0.13"
    required_checks: Mapping[str, bool | int] = field(
        default_factory=lambda: dict(_CHECKS)
    )

    def validate_dependencies(self) -> None:
        result = materialize_module._validate_swe_gym_harness(self.harness_root)
        if result is None or set(result) != {"constants", "log_parsers", "grading"}:
            raise ValueError("pinned SWE-Gym harness is incomplete")

    def validate_candidate(self, row: Mapping[str, Any]) -> None:
        verifier = _mapping(row, "verifier")
        if verifier.get("kind") != "swebench-harness-v1":
            raise semantic_admission.QuarantineTask(
                "unsupported_verifier_kind",
                "SWE-Gym verifier kind is unsupported",
            )
        fail_to_pass = _string_list(
            verifier.get("fail_to_pass"),
            "fail_to_pass",
        )
        pass_to_pass = _string_list(
            verifier.get("pass_to_pass"),
            "pass_to_pass",
        )
        if not fail_to_pass:
            raise semantic_admission.QuarantineTask(
                "fail_to_pass_missing",
                "SWE-Gym has no FAIL_TO_PASS test",
            )
        if len(set(fail_to_pass + pass_to_pass)) != len(
            fail_to_pass + pass_to_pass
        ):
            raise semantic_admission.QuarantineTask(
                "duplicate_test_inventory",
                "SWE-Gym expected test IDs are not unique",
            )
        _text(row, "repo")
        _text(_mapping(row, "source_metadata"), "version")

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
            swe_rebench_log_parsers=None,
            swe_rebench_constants=None,
            swe_rebench_eval=None,
            swe_rebench_admission_manifest=None,
            swe_gym_harness_root=self.harness_root,
            swe_gym_admission_manifest=None,
            allow_mutable_images=False,
            allow_unadmitted_r2e_dry_run=False,
            allow_unadmitted_swe_rebench_dry_run=False,
            allow_unadmitted_swe_gym_dry_run=True,
            limit=None,
            summary=None,
        )

    def validate_materialized(
        self,
        task_dir: Path,
        row: Mapping[str, Any],
    ) -> None:
        verifier = _mapping(row, "verifier")
        config = _read_json(task_dir / "tests" / "verifier_config.json")
        expected = {
            "instance_id": _text(row, "instance_id"),
            "repo": _text(row, "repo").lower(),
            "version": _text(_mapping(row, "source_metadata"), "version"),
            "base_commit": _text(row, "base_commit"),
            "fail_to_pass": _string_list(
                verifier.get("fail_to_pass"),
                "fail_to_pass",
            ),
            "pass_to_pass": _string_list(
                verifier.get("pass_to_pass"),
                "pass_to_pass",
            ),
            "harness_commit": materialize_module._SWE_GYM_HARNESS_COMMIT,
            "harness_version": materialize_module._SWE_GYM_HARNESS_VERSION,
            "constants_sha256": materialize_module._SWE_GYM_CONSTANTS_SHA256,
            "log_parsers_sha256": materialize_module._SWE_GYM_LOG_PARSERS_SHA256,
            "grading_sha256": materialize_module._SWE_GYM_GRADING_SHA256,
            "harbor_adapter_commit": materialize_module._SWE_GYM_HARBOR_COMMIT,
            "harbor_adapter_sha256": materialize_module._SWE_GYM_HARBOR_ADAPTER_SHA256,
            "source_image": _source_image(row),
        }
        if config != expected:
            raise semantic_admission.QuarantineTask(
                "official_verifier_binding_mismatch",
                "materialized SWE-Gym verifier differs from pinned inputs",
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
        if report.get("kind") != self.report_kind:
            raise semantic_admission.SystemicAdmissionError(
                "SWE-Gym report did not use the pinned official parser"
            )

    def admission_metadata(self, row: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "dataset_revision": materialize_module._SWE_GYM_DATASET_REVISION,
            "harness_commit": materialize_module._SWE_GYM_HARNESS_COMMIT,
            "harness_version": materialize_module._SWE_GYM_HARNESS_VERSION,
            "constants_sha256": materialize_module._SWE_GYM_CONSTANTS_SHA256,
            "log_parsers_sha256": materialize_module._SWE_GYM_LOG_PARSERS_SHA256,
            "grading_sha256": materialize_module._SWE_GYM_GRADING_SHA256,
            "harbor_adapter_commit": materialize_module._SWE_GYM_HARBOR_COMMIT,
            "harbor_adapter_sha256": materialize_module._SWE_GYM_HARBOR_ADAPTER_SHA256,
        }


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


def _source_image(row: Mapping[str, Any]) -> str:
    return _text(_mapping(row, "sandbox"), "source_image")


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
    parser.add_argument("--swe-gym-harness-root", type=Path, required=True)
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
    adapter = SweGymAdapter(harness_root=args.swe_gym_harness_root)
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
