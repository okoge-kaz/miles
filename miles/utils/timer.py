import gc
import logging
from contextlib import contextmanager
from functools import wraps
from threading import get_ident
from time import monotonic, thread_time, time

import torch.distributed

from .misc import SingletonMeta

__all__ = ["Timer", "measure_time", "timer"]

logger = logging.getLogger(__name__)


class Timer(metaclass=SingletonMeta):
    def __init__(self):
        self.timers = {}
        self.start_time = {}
        # observers called sink(name, t0, t1) on end(); an optional
        # sink.begin(name, t0) also hears starts. Registered by the dashboard.
        self.event_sinks = []

    def start(self, name):
        assert name not in self.start_time, f"Timer {name} already started."
        self.start_time[name] = time()
        for sink in self.event_sinks:
            begin = getattr(sink, "begin", None)
            if callable(begin):
                begin(name, self.start_time[name])
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.info(f"Timer {name} start")

    def end(self, name):
        assert name in self.start_time, f"Timer {name} not started."
        start_time = self.start_time[name]
        elapsed_time = time() - start_time
        for sink in self.event_sinks:
            sink(name, start_time, start_time + elapsed_time)
        self.add(name, elapsed_time)
        del self.start_time[name]
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.info(f"Timer {name} end (elapsed: {elapsed_time:.1f}s)")

    def reset(self, name=None):
        if name is None:
            self.timers = {}
        elif name in self.timers:
            del self.timers[name]

    def add(self, name, elapsed_time):
        self.timers[name] = self.timers.get(name, 0) + elapsed_time

    def log_dict(self):
        return self.timers

    @contextmanager
    def context(self, name):
        self.start(name)
        try:
            yield
        finally:
            self.end(name)


def timer(name_or_func):
    """
    Can be used either as a decorator or a context manager:

    @timer
    def func():
        ...

    or

    with timer("block_name"):
        ...
    """
    # When used as a context manager
    if isinstance(name_or_func, str):
        name = name_or_func
        return Timer().context(name)

    func = name_or_func

    @wraps(func)
    def wrapper(*args, **kwargs):
        with Timer().context(func.__name__):
            return func(*args, **kwargs)

    return wrapper


@contextmanager
def inverse_timer(name):
    Timer().end(name)
    try:
        yield
    finally:
        Timer().start(name)


@contextmanager
def measure_time(metrics: dict | None, name: str):
    """Measure a synchronous block's wall, thread CPU, and GC time; do not span an await.

    CPU and GC durations overlap wall time. Passing None disables measurement.
    GC callbacks are scoped to this block and ignore collections on other threads.
    """
    if metrics is None:
        yield
        return

    owner = get_ident()
    gc_started = None
    gc_seconds = 0.0
    gc_count = 0

    def record_gc(phase, info):
        nonlocal gc_started, gc_seconds, gc_count
        if get_ident() != owner:
            return
        if phase == "start":
            gc_started = monotonic()
        elif phase == "stop" and gc_started is not None:
            gc_seconds += monotonic() - gc_started
            gc_count += 1
            gc_started = None

    started, cpu_started = monotonic(), thread_time()
    gc.callbacks.append(record_gc)
    try:
        yield
    finally:
        gc.callbacks.remove(record_gc)
        metrics[f"{name}/wall_time"] = monotonic() - started
        metrics[f"{name}/cpu_time"] = thread_time() - cpu_started
        metrics[f"{name}/gc_time"] = gc_seconds
        metrics[f"{name}/gc_collections"] = gc_count
