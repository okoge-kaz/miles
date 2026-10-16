import argparse
import multiprocessing
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager


def _positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def add_parallel_arguments(parser):
    parser.add_argument(
        "--num-proc", type=_positive_int, default=1, help="Conversion workers; 1 runs in the main process"
    )
    parser.add_argument("--chunk-size", type=_positive_int, default=128, help="Rows per worker batch (default: 128)")


def _map_batch(adapter, batch):
    results = []
    for row in batch:
        try:
            results.append(adapter(row))
        except Exception as exc:
            # Preserve the valid prefix, including when a caller stops at --limit.
            return results, exc
    return results, None


def _ordered_results(executor, adapter, rows, *, num_proc, chunk_size):
    rows = iter(rows)
    pending = deque()
    finished = False
    source_error = None
    while pending or not finished:
        while len(pending) < 2 * num_proc and not finished:
            batch = []
            try:
                for _ in range(chunk_size):
                    batch.append(next(rows))
            except StopIteration:
                finished = True
            except Exception as exc:
                source_error = exc
                finished = True
            if batch:
                pending.append(executor.submit(_map_batch, adapter, batch))
        if pending:
            results, error = pending.popleft().result()
            yield from results
            if error is not None:
                raise error
    if source_error is not None:
        raise source_error


@contextmanager
def map_rows(rows, adapter, *, num_proc=1, chunk_size=128):
    """Map in input order with bounded prefetch; always close workers on early exit."""
    if num_proc < 1 or chunk_size < 1:
        raise ValueError("num_proc and chunk_size must be positive")
    if num_proc == 1:
        yield map(adapter, rows)
        return
    # Do not fork an already initialized Arrow/native thread pool.
    executor = ProcessPoolExecutor(max_workers=num_proc, mp_context=multiprocessing.get_context("spawn"))
    try:
        yield _ordered_results(executor, adapter, rows, num_proc=num_proc, chunk_size=chunk_size)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
