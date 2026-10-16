import base64
import json

import pytest

from experiments.src.training.multi_env_preflight import check_reward_runtime, read_sources


def test_preflight_resolves_relative_sources_and_rejects_raw_or_eval_inputs(tmp_path):
    data = tmp_path / "rows.jsonl"
    config = tmp_path / "mix.json"
    source = {"name": "math", "path": "rows.jsonl"}
    config.write_text(json.dumps({"datasets": [source]}))
    row = {"prompt": "Question", "label": "42"}
    data.write_text("\n" + json.dumps(row) + "\n")
    assert read_sources(config) == [(source, row)]
    data.write_text(json.dumps({"prompt": "raw instruction data"}) + "\n")
    with pytest.raises(ValueError, match="converted RL label"):
        read_sources(config)
    data.write_text(json.dumps({**row, "metadata": {"eval_only": True}}) + "\n")
    with pytest.raises(ValueError, match="Evaluation-only"):
        read_sources(config)


def test_preflight_stops_when_code_sandbox_cannot_score_known_answers(monkeypatch):
    from experiments.src.nemo_gym.bridge import reward
    from experiments.src.training import multi_env_preflight

    async def unscored(*args):
        return [None, None]

    monkeypatch.setattr(reward, "reward", unscored)
    monkeypatch.setattr(multi_env_preflight, "check_nemo_service", lambda service: None)
    source = {"custom_rm_path": "experiments.src.nemo_gym.bridge.reward.reward"}
    metadata = {"verifier": "python_code", "unit_tests": {"inputs": ["2"], "outputs": ["3"]}}
    with pytest.raises(RuntimeError, match="known-answer probe"):
        check_reward_runtime([(source, {"metadata": metadata})])


def test_inline_preflight_and_host_mount_validation(tmp_path):
    from experiments.common.dataset_args import check_prompt_paths

    path = tmp_path / "rows.jsonl"
    row = {"prompt": "Question", "label": "42"}
    path.write_text(json.dumps(row) + "\n")

    def encode(source):
        return "base64:" + base64.b64encode(json.dumps({"datasets": [source]}).encode()).decode()

    source = {"name": "math", "path": str(path)}
    assert read_sources(encode(source)) == [(source, row)]
    assert read_sources("", files=[str(path)]) == [({"path": str(path)}, row)]
    environment = {"PROMPT_DATA_CONFIG": encode({**source, "path": "/data/rows.jsonl"})}
    check_prompt_paths(environment, data_root=tmp_path, repo_root=tmp_path)
    path.unlink()
    with pytest.raises(ValueError, match="Missing prompt source"):
        check_prompt_paths(environment, data_root=tmp_path, repo_root=tmp_path)
