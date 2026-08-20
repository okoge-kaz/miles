from types import SimpleNamespace

from miles.utils.tracking_utils import wandb_utils


def _args(**overrides):
    values = {
        "env_report": None,
        "rank": 0,
        "sglang_enable_metrics": False,
        "use_wandb": True,
        "wandb_dir": None,
        "wandb_group": "group",
        "wandb_host": None,
        "wandb_key": None,
        "wandb_mode": None,
        "wandb_project": "project",
        "wandb_random_suffix": False,
        "wandb_run_id": None,
        "wandb_team": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_primary_wandb_init_uses_extended_init_timeout(monkeypatch):
    init_calls = []

    monkeypatch.setattr(wandb_utils.wandb, "init", lambda **kwargs: init_calls.append(kwargs))
    monkeypatch.setattr(wandb_utils.wandb, "define_metric", lambda *args, **kwargs: None)
    monkeypatch.setattr(wandb_utils.wandb, "run", SimpleNamespace(id="run-id"), raising=False)

    args = _args()
    wandb_utils.init_wandb_primary(args)

    settings = init_calls[0]["settings"]
    assert settings.mode == "shared"
    assert settings.x_primary is True
    assert settings.init_timeout == 300.0
    assert args.wandb_run_id == "run-id"


def test_secondary_wandb_init_uses_extended_init_timeout(monkeypatch):
    init_calls = []

    monkeypatch.setattr(wandb_utils.wandb, "init", lambda **kwargs: init_calls.append(kwargs))
    monkeypatch.setattr(wandb_utils.wandb, "define_metric", lambda *args, **kwargs: None)

    args = _args(wandb_run_id="run-id")
    wandb_utils.init_wandb_secondary(args)

    settings = init_calls[0]["settings"]
    assert settings.mode == "shared"
    assert settings.x_primary is False
    assert settings.x_update_finish_state is False
    assert settings.init_timeout == 300.0


def test_metric_namespaces_use_their_declared_step_axes(monkeypatch):
    calls = []
    monkeypatch.setattr(
        wandb_utils.wandb,
        "define_metric",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    wandb_utils._init_wandb_common()

    for step_metric, prefixes in wandb_utils._STEP_METRIC_PREFIXES.items():
        assert ((step_metric,), {}) in calls
        for prefix in prefixes:
            assert ((f"{prefix}/*",), {"step_metric": step_metric}) in calls
