import asyncio
import importlib.util
import unittest
from unittest.mock import AsyncMock, Mock, patch

from experiments.src.difficulty_filtering.core.config import FilterConfig
from experiments.src.difficulty_filtering.core.measure import generate_request, measure_one


@unittest.skipUnless(importlib.util.find_spec("aiohttp"), "HTTP retry check requires the runtime image")
class RetryTest(unittest.TestCase):
    def test_connection_failure_retries_without_committing_partial_group(self):
        import aiohttp

        output = {"text": "answer", "meta_info": {"finish_reason": "stop", "completion_tokens": 2}}
        response = Mock()
        response.json = AsyncMock(return_value=output)
        context = AsyncMock()
        context.__aenter__.return_value = response
        session = Mock()
        session.post.side_effect = [aiohttp.ServerDisconnectedError(), context]
        with patch("asyncio.sleep", new_callable=AsyncMock):
            actual = asyncio.run(generate_request(session, "http://test", [1, 2], {}, attempts=2))
        self.assertEqual(actual, output)
        self.assertEqual(session.post.call_count, 2)
        for call in session.post.call_args_list:
            self.assertEqual(call.args[0], "http://test/generate")
            self.assertEqual(call.kwargs["json"], {"input_ids": [1, 2], "sampling_params": {}})

    def test_rl_single_response_sampling_across_independently_seeded_engines(self):
        config = FilterConfig(
            model_path="/unused",
            prompt_data="/unused",
            measurement_dir="/unused",
            filtered_output="/unused-output",
            policy="test",
            num_engines=8,
        )
        tokenizer = Mock()
        tokenizer.encode.return_value = [1]
        output = {"text": "answer", "meta_info": {"finish_reason": "stop", "completion_tokens": 2}}
        with (
            patch("experiments.src.difficulty_filtering.core.measure.generate_request", new_callable=AsyncMock) as generate,
            patch("experiments.src.difficulty_filtering.core.measure.score_responses", new_callable=AsyncMock) as score,
        ):
            generate.return_value = output
            score.return_value = [0.0] * 8 + [1.0] * 8
            urls = [f"http://engine-{index}" for index in range(config.num_engines)]
            for index in [0, 7, 8, 15]:
                generate.reset_mock()
                record, _ = asyncio.run(
                    measure_one(Mock(), urls, config, tokenizer, index, {"prompt": "q", "label": "7"})
                )
                self.assertEqual(record["n_correct"], 8)
                self.assertEqual(generate.await_count, 16)
                self.assertEqual([call.args[1] for call in generate.call_args_list], urls * 2)
                for call in generate.call_args_list:
                    self.assertEqual(call.args[2], [1])
                    self.assertNotIn("sampling_seed", call.args[3])
                    self.assertNotIn("n", call.args[3])
                    self.assertFalse(call.kwargs)

            # Compare the production RL sampling helper, including decode/stop defaults.
            from types import SimpleNamespace

            from miles.rollout.inference_rollout.inference_rollout_common import compute_sampling_params

            expected = compute_sampling_params(
                SimpleNamespace(rollout_stop=None, rollout_stop_token_ids=None, rollout_skip_special_tokens=False),
                temperature=1.0,
                top_p=1.0,
                top_k=-1,
                max_new_tokens=16384,
            )
            self.assertEqual(generate.call_args.args[3], expected)

    def test_connection_failure_exhaustion_is_fatal(self):
        import aiohttp

        session = Mock()
        session.post.side_effect = aiohttp.ServerDisconnectedError()
        with patch("asyncio.sleep", new_callable=AsyncMock), self.assertRaises(aiohttp.ServerDisconnectedError):
            asyncio.run(generate_request(session, "http://test", [1], {}, attempts=2))
        self.assertEqual(session.post.call_count, 2)
