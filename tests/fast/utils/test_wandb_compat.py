from types import SimpleNamespace

from miles.utils.tracking_utils import wandb_utils


def test_primary_init_generates_random_suffix_without_wandb_util(monkeypatch):
    init_calls = []
    monkeypatch.setattr(wandb_utils.secrets, "token_hex", lambda _: "1234abcd")
    monkeypatch.setattr(wandb_utils.wandb, "init", lambda **kwargs: init_calls.append(kwargs))
    monkeypatch.setattr(wandb_utils.wandb, "define_metric", lambda *args, **kwargs: None)
    monkeypatch.setattr(wandb_utils.wandb, "run", SimpleNamespace(id="run-id"), raising=False)

    args = SimpleNamespace(
        env_report=None,
        rank=0,
        sglang_enable_metrics=False,
        use_wandb=True,
        wandb_dir=None,
        wandb_group="group",
        wandb_host=None,
        wandb_key=None,
        wandb_mode=None,
        wandb_project="project",
        wandb_random_suffix=True,
        wandb_run_id=None,
        wandb_team=None,
    )
    wandb_utils.init_wandb_primary(args)

    assert init_calls[0]["group"] == "group_1234abcd"
    assert init_calls[0]["name"] == "group_1234abcd-RANK_0"
