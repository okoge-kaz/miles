from unittest.mock import patch

from tests.fast.ray.rollout.conftest import fake_actor_handle, make_dataclass_group


@patch("miles.ray.rollout.server_group.ray.kill")
@patch("miles.ray.rollout.server_group.ray.get", side_effect=RuntimeError("shutdown failed"))
def test_stop_engines_forces_actor_termination_after_shutdown_failure(mock_ray_get, mock_ray_kill):
    group = make_dataclass_group(num_engines=1)
    actor_handle = fake_actor_handle()
    group.all_engines[0].mark_allocated_uninitialized(actor_handle)

    group.stop_engines(engine_indices=[0])

    mock_ray_get.assert_called_once()
    mock_ray_kill.assert_called_once_with(actor_handle)
    assert not group.all_engines[0].is_allocated
