"""Live admission for hardened-local SWE-bench Verified evaluation tasks."""

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

ADMISSION_SCHEMA = "miles-swebench-verified-hardened-local-admission-v1"
_DATASET_ID = "princeton-nlp/SWE-bench_Verified"


@dataclass(frozen=True)
class SwebenchVerifiedAdapter:
    """Eval-only adapter with an official parser and hardened local execution."""

    harness_root: Path
    source_schema: str = "swebench"
    admission_schema: str = ADMISSION_SCHEMA
    checkpoint_label: str = "swebench-verified-admit"
    report_kind: str = "swebench-verified-hardened-local-v2.0.13"
    required_checks: Mapping[str, bool | int] = field(
        default_factory=lambda: dict(
            materialize_module._REPOSITORY_ADMISSION_CHECKS
        )
    )

    def validate_dependencies(self) -> None:
        result = materialize_module._validate_swebench_harness(self.harness_root)
        if result is None or set(result) != {"constants", "log_parsers", "grading"}:
            raise ValueError("pinned official SWE-bench harness is incomplete")

    def validate_candidate(self, row: Mapping[str, Any]) -> None:
        if row.get("source_dataset") != _DATASET_ID or row.get("eval_only") is not True:
            raise semantic_admission.QuarantineTask(
                "invalid_evaluation_provenance",
                "SWE-bench Verified admission accepts only the pinned eval-only dataset",
            )
        source_metadata = row.get("source_metadata")
        if not isinstance(source_metadata, dict) or source_metadata.get("split") != "test":
            raise semantic_admission.QuarantineTask(
                "invalid_evaluation_split",
                "SWE-bench Verified admission accepts only the official test split",
            )
        verifier = _mapping(row, "verifier")
        if verifier.get("kind") != "swebench-harness-v1":
            raise semantic_admission.QuarantineTask(
                "unsupported_verifier_kind",
                "SWE-bench Verified verifier kind is unsupported",
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
                "SWE-bench Verified has no FAIL_TO_PASS test",
            )
        if len(set(fail_to_pass + pass_to_pass)) != len(
            fail_to_pass + pass_to_pass
        ):
            raise semantic_admission.QuarantineTask(
                "duplicate_test_inventory",
                "SWE-bench Verified expected test IDs are not unique",
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
            swe_gym_harness_root=None,
            swe_gym_admission_manifest=None,
            swebench_harness_root=self.harness_root,
            swebench_verified_admission_manifest=None,
            allow_mutable_images=False,
            allow_unadmitted_r2e_dry_run=False,
            allow_unadmitted_swe_rebench_dry_run=False,
            allow_unadmitted_swe_gym_dry_run=False,
            allow_unadmitted_swebench_verified_dry_run=True,
            limit=None,
            summary=None,
        )

    def validate_materialized(
        self,
        task_dir: Path,
        row: Mapping[str, Any],
    ) -> None:
        verifier = row.get("verifier")
        source_metadata = row.get("source_metadata")
        if not isinstance(verifier, dict) or not isinstance(source_metadata, dict):
            raise semantic_admission.QuarantineTask(
                "required_metadata_missing",
                "Verified verifier metadata is absent",
            )
        config = json.loads(
            (task_dir / "tests" / "verifier_config.json").read_text(
                encoding="utf-8"
            )
        )
        expected = {
            "instance_id": row["instance_id"],
            "repo": str(row["repo"]).lower(),
            "version": source_metadata["version"],
            "base_commit": row["base_commit"],
            "fail_to_pass": verifier["fail_to_pass"],
            "pass_to_pass": verifier["pass_to_pass"],
            "harness_repository": materialize_module._SWEBENCH_HARNESS_REPOSITORY,
            "harness_commit": materialize_module._SWEBENCH_HARNESS_COMMIT,
            "harness_version": materialize_module._SWEBENCH_HARNESS_VERSION,
            "constants_sha256": materialize_module._SWEBENCH_CONSTANTS_SHA256,
            "log_parsers_sha256": (
                materialize_module._SWEBENCH_LOG_PARSERS_SHA256
            ),
            "grading_sha256": materialize_module._SWEBENCH_GRADING_SHA256,
            "score_semantics": (
                materialize_module._SWEBENCH_HARDENED_SCORE_SEMANTICS
            ),
            "source_image": row["sandbox"]["source_image"],
        }
        if config != expected:
            raise semantic_admission.QuarantineTask(
                "official_verifier_binding_mismatch",
                "materialized Verified verifier differs from pinned official inputs",
            )
        if (task_dir / "tests" / "test_patch.diff").read_text(
            encoding="utf-8"
        ) != verifier.get("test_patch"):
            raise semantic_admission.QuarantineTask(
                "private_test_binding_mismatch",
                "materialized Verified hidden patch differs from the private record",
            )
        policy = json.loads(
            (task_dir / "tests" / "model_path_policy.json").read_text(
                encoding="utf-8"
            )
        )
        if (
            not isinstance(policy, dict)
            or policy.get("schema_version")
            != materialize_module._EVAL_MODEL_PATH_POLICY_SCHEMA
            or policy.get("policy_mode") != "deny-sensitive-paths"
        ):
            raise semantic_admission.QuarantineTask(
                "evaluation_path_policy_mismatch",
                "SWE-bench Verified did not materialize the eval-safe path policy",
            )

    def validate_report(
        self,
        row: Mapping[str, Any],
        report: Mapping[str, Any],
        reward: int,
    ) -> None:
        if reward == 0 and "kind" not in report:
            return
        if (
            report.get("kind") != self.report_kind
            or report.get("score_semantics")
            != materialize_module._SWEBENCH_HARDENED_SCORE_SEMANTICS
            or report.get("official_harness_commit")
            != materialize_module._SWEBENCH_HARNESS_COMMIT
        ):
            raise semantic_admission.SystemicAdmissionError(
                "Verified report did not use the pinned official parser/hardened semantics"
            )

    def admission_metadata(self, row: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "dataset_revision": materialize_module._SWEBENCH_VERIFIED_DATASET_REVISION,
            "harness_repository": materialize_module._SWEBENCH_HARNESS_REPOSITORY,
            "harness_commit": materialize_module._SWEBENCH_HARNESS_COMMIT,
            "harness_version": materialize_module._SWEBENCH_HARNESS_VERSION,
            "constants_sha256": materialize_module._SWEBENCH_CONSTANTS_SHA256,
            "log_parsers_sha256": materialize_module._SWEBENCH_LOG_PARSERS_SHA256,
            "grading_sha256": materialize_module._SWEBENCH_GRADING_SHA256,
            "score_semantics": materialize_module._SWEBENCH_HARDENED_SCORE_SEMANTICS,
            "harbor_adapter_commit": materialize_module._SWE_GYM_HARBOR_COMMIT,
            "harbor_adapter_sha256": (
                materialize_module._SWE_GYM_HARBOR_ADAPTER_SHA256
            ),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-manifest", type=Path, required=True)
    parser.add_argument("--admitted-manifest", type=Path, required=True)
    parser.add_argument("--admission-manifest", type=Path, required=True)
    parser.add_argument("--quarantine-manifest", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--swebench-harness-root", type=Path, required=True)
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
    adapter = SwebenchVerifiedAdapter(harness_root=args.swebench_harness_root)
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
