from __future__ import annotations

import pytest

from miles.rollout.harbor.environment_config import (
    build_harbor_environment_config,
    environment_uses_local_docker,
    get_harbor_environment_spec,
)


def test_docker_defaults_preserve_existing_behavior() -> None:
    spec = get_harbor_environment_spec({})

    assert spec.environment_type == "docker"
    assert spec.delete is False
    assert spec.override_memory_mb is None
    assert environment_uses_local_docker({}) is True


def test_docker_parses_memory_delete_and_allowed_hosts() -> None:
    spec = get_harbor_environment_spec(
        {
            "HARBOR_ENV_TYPE": "DOCKER",
            "HARBOR_DELETE_CONTAINERS": "true",
            "HARBOR_OVERRIDE_MEMORY_MB": "8192",
            "HARBOR_ENV_ALLOWED_HOSTS": "router.internal, pypi.org\ngithub.com",
        }
    )

    assert spec.delete is True
    assert spec.override_memory_mb == 8192
    assert spec.extra_allowed_hosts == ("router.internal", "pypi.org", "github.com")


def test_daytona_preserves_storage_snapshot_and_keep_semantics() -> None:
    spec = get_harbor_environment_spec(
        {
            "HARBOR_ENV_TYPE": "daytona",
            "HARBOR_DAYTONA_DISK_GB": "20",
            "HARBOR_DAYTONA_AUTO_SNAPSHOT": "1",
            "HARBOR_KEEP_SANDBOX": "true",
        }
    )

    assert spec.environment_type == "daytona"
    assert spec.delete is False
    assert spec.override_storage_mb == 20 * 1024
    assert dict(spec.provider_kwargs) == {"auto_snapshot": True}
    assert environment_uses_local_docker({"HARBOR_ENV_TYPE": "daytona"}) is False


def test_e2b_is_ephemeral_and_never_embeds_the_api_key() -> None:
    secret = "e2b_test_secret"
    spec = get_harbor_environment_spec(
        {
            "HARBOR_ENV_TYPE": "e2b",
            "E2B_API_KEY": secret,
            "HARBOR_OVERRIDE_MEMORY_MB": "16384",
            "HARBOR_ENV_ALLOWED_HOSTS": "router.internal",
        }
    )

    kwargs = spec.as_harbor_kwargs()
    assert spec.environment_type == "e2b"
    assert spec.delete is True
    assert kwargs["override_memory_mb"] == 16384
    assert kwargs["extra_allowed_hosts"] == ["router.internal"]
    assert secret not in repr(spec)
    assert secret not in repr(kwargs)
    assert environment_uses_local_docker({"HARBOR_ENV_TYPE": "e2b"}) is False


def test_e2b_fails_closed_without_a_process_environment_key() -> None:
    with pytest.raises(ValueError, match="E2B_API_KEY"):
        get_harbor_environment_spec({"HARBOR_ENV_TYPE": "e2b"})


def test_e2b_rejects_keep_sandbox_instead_of_promising_false_persistence() -> None:
    with pytest.raises(ValueError, match="HARBOR_KEEP_SANDBOX"):
        get_harbor_environment_spec(
            {
                "HARBOR_ENV_TYPE": "e2b",
                "E2B_API_KEY": "present",
                "HARBOR_KEEP_SANDBOX": "1",
            }
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("HARBOR_OVERRIDE_MEMORY_MB", "0"),
        ("HARBOR_OVERRIDE_MEMORY_MB", "not-an-int"),
        ("HARBOR_DAYTONA_DISK_GB", "-1"),
    ],
)
def test_positive_integer_settings_are_validated(name: str, value: str) -> None:
    environ = {"HARBOR_ENV_TYPE": "daytona", name: value}
    with pytest.raises(ValueError, match=name):
        get_harbor_environment_spec(environ)


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported HARBOR_ENV_TYPE"):
        get_harbor_environment_spec({"HARBOR_ENV_TYPE": "unknown"})


def test_builder_keeps_harbor_an_optional_dependency() -> None:
    captured: dict[str, object] = {}

    def fake_config(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return captured

    result = build_harbor_environment_config(
        {"HARBOR_ENV_TYPE": "e2b", "E2B_API_KEY": "present"},
        config_factory=fake_config,
    )

    assert result is captured
    assert captured == {
        "type": "e2b",
        "delete": True,
        "override_memory_mb": None,
    }
