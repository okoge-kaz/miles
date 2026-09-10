from miles.utils.misc import checkpoint_artifacts_due


def _due(rollout_id, **kwargs):
    kwargs.setdefault("save_interval", 20)
    kwargs.setdefault("hf_save_interval", None)
    return checkpoint_artifacts_due(rollout_id, **kwargs)


class TestCheckpointArtifactsDue:
    def test_hf_follows_save_interval_when_unset(self):
        """The default has to reproduce the pre-`--hf-save-interval` behaviour exactly."""
        for rollout_id in range(60):
            write_dist, write_hf = _due(rollout_id)
            assert write_dist == write_hf

    def test_denser_hf_cadence_does_not_pull_the_dist_checkpoint_with_it(self):
        # rollout_id is 0-based, so step 5 is rollout_id 4.
        assert _due(4, save_interval=20, hf_save_interval=5) == (False, True)
        assert _due(9, save_interval=20, hf_save_interval=5) == (False, True)
        assert _due(19, save_interval=20, hf_save_interval=5) == (True, True)

    def test_neither_is_due_between_cadences(self):
        assert _due(3, save_interval=20, hf_save_interval=5) == (False, False)

    def test_hf_only_when_there_is_no_dist_checkpoint_at_all(self):
        assert _due(4, save_interval=None, hf_save_interval=5) == (False, True)

    def test_external_trigger_forces_the_resumable_checkpoint_only(self):
        """The sentinel exists to make a run resumable on demand, not to export weights."""
        assert _due(3, save_interval=20, hf_save_interval=5, external_save=True) == (True, False)

    def test_external_trigger_still_writes_hf_when_the_cadence_is_shared(self):
        assert _due(3, save_interval=20, external_save=True) == (True, True)

    def test_final_rollout_writes_both(self):
        assert _due(99, save_interval=20, hf_save_interval=5, num_rollout=100) == (True, True)

    def test_epoch_boundary_writes_both(self):
        assert _due(31, save_interval=20, hf_save_interval=5, num_rollout_per_epoch=32) == (True, True)
