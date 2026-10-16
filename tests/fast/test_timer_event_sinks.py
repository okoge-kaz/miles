import gc
import time

import pytest

from miles.utils.timer import Timer, inverse_timer, measure_time, timer


@pytest.fixture
def fresh_timer():
    instance = Timer()
    saved = (instance.timers, instance.start_time, instance.event_sinks)
    instance.timers, instance.start_time, instance.event_sinks = {}, {}, []
    yield instance
    instance.timers, instance.start_time, instance.event_sinks = saved


def test_sink_receives_interval(fresh_timer):
    events = []
    fresh_timer.event_sinks.append(lambda name, t0, t1: events.append((name, t0, t1)))

    before = time.time()
    with timer("phase_a"):
        time.sleep(0.01)
    after = time.time()

    [(name, t0, t1)] = events
    assert name == "phase_a"
    assert before <= t0 <= t1 <= after
    assert t1 - t0 == pytest.approx(fresh_timer.timers["phase_a"])


def test_no_sinks_keeps_behavior_identical(fresh_timer):
    with timer("phase_b"):
        pass
    assert list(fresh_timer.timers) == ["phase_b"]
    assert fresh_timer.start_time == {}


def test_multiple_sinks_and_nested_timers(fresh_timer):
    events = []
    fresh_timer.event_sinks.append(lambda *e: events.append(("s1", e[0])))
    fresh_timer.event_sinks.append(lambda *e: events.append(("s2", e[0])))

    fresh_timer.start("outer")
    with inverse_timer("outer"):
        with timer("inner"):
            pass
    fresh_timer.end("outer")

    tags = [tag for tag, _ in events]
    assert tags == ["s1", "s2"] * 3  # every end() fans out to every sink, in order
    timer_names = [name for _, name in events]
    # inverse_timer ends "outer" first, "inner" ends inside, "outer" ends last
    assert timer_names == ["outer", "outer", "inner", "inner", "outer", "outer"]


def test_sink_with_begin_hears_starts():
    timer = Timer()
    begins, ends = [], []

    class Sink:
        def begin(self, name, t0):
            begins.append((name, t0))

        def __call__(self, name, t0, t1):
            ends.append(name)

    timer.event_sinks.append(Sink())
    try:
        timer.start("long_phase")
        assert [name for name, _ in begins] == ["long_phase"]  # visible while open
        assert ends == []
        timer.end("long_phase")
        assert ends == ["long_phase"]
    finally:
        timer.event_sinks.clear()
        timer.reset()


def test_block_measurement_cleans_up_gc_callback_on_failure():
    callbacks = list(gc.callbacks)
    metrics = {}
    with pytest.raises(RuntimeError, match="block failed"):
        with measure_time(metrics, "stage"):
            gc.collect()
            raise RuntimeError("block failed")

    assert gc.callbacks == callbacks
    assert metrics["stage/wall_time"] >= metrics["stage/gc_time"] >= 0
    assert metrics["stage/cpu_time"] >= 0
    assert metrics["stage/gc_collections"] >= 1


def test_disabled_block_measurement_does_not_read_clocks_or_register_callbacks(monkeypatch):
    import miles.utils.timer as timer_module

    def fail():
        pytest.fail("disabled measurement must not read a clock")

    monkeypatch.setattr(timer_module, "monotonic", fail)
    monkeypatch.setattr(timer_module, "thread_time", fail)
    callbacks = list(gc.callbacks)
    with measure_time(None, "stage"):
        assert gc.callbacks == callbacks
