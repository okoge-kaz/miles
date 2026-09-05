"""Preserve scheduler-authored provenance in SGLang's synthetic abort replies."""

from functools import wraps

_STATE_FIELD = "_miles_policy_version_meta_info"


def _patch_tokenizer_manager(manager_class: type) -> None:
    if getattr(manager_class, "_miles_abort_provenance_installed", False):
        return
    handle_batch_output = manager_class._handle_batch_output
    handle_abort_req = manager_class._handle_abort_req

    @wraps(handle_batch_output)
    async def remember_policy_provenance(self, recv_obj):
        await handle_batch_output(self, recv_obj)
        # The original handler unpickles these scheduler stats and updates the
        # accumulated output prefix. Cache them even when out_list is consumed.
        stats = recv_obj.time_stats
        completion_tokens = getattr(recv_obj, "completion_tokens", None)
        if stats is None or completion_tokens is None:
            return
        for rid, time_stats, token_count in zip(recv_obj.rids, stats, completion_tokens, strict=True):
            state = self.rid_to_state.get(rid)
            if state is not None:
                setattr(state, _STATE_FIELD, time_stats.convert_to_policy_version_meta_info(token_count))

    @wraps(handle_abort_req)
    def preserve_abort_provenance(self, recv_obj):
        state = self.rid_to_state.get(recv_obj.rid)
        handle_abort_req(self, recv_obj)
        cached = getattr(state, _STATE_FIELD, None)
        if not cached or not state.out_list:
            return
        meta_info = state.out_list[-1]["meta_info"]
        if meta_info.get("finish_reason", {}).get("type") != "abort":
            return
        # Never mix a newer runtime's own abort provenance with cached fields.
        if any(key in meta_info for key in cached):
            return
        token_count = meta_info["completion_tokens"]
        for key, value in cached.items():
            if key == "response_weight_version_segments":
                value = [
                    [start, min(end, token_count), version] for start, end, version in value if start < token_count
                ]
            meta_info[key] = value

    manager_class._handle_batch_output = remember_policy_provenance
    manager_class._handle_abort_req = preserve_abort_provenance
    manager_class._miles_abort_provenance_installed = True


def install_abort_policy_provenance() -> None:
    # Import only in the spawned HTTP server; tokenizer initialization imports
    # the inference runtime and must not run in CPU-only replay audit processes.
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

    _patch_tokenizer_manager(TokenizerManager)
