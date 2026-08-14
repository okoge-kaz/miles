"""Local dense retriever for the Search-R1 recipe: FAISS over wiki-18, e5 queries.

Serves the one endpoint `examples/experimental/search-r1/local_search_server.py`
calls, so the rollout never leaves the node:

    POST /retrieve  {"queries": [...], "topk": 3}  ->  {"result": [[{document:{contents}}]]}
    GET  /health

Written here rather than reusing Search-R1's own server because that one pulls
in the whole Search-R1 package (and its verl dependency) for what is a FAISS
lookup plus one encoder forward. The contract is small and stable; the
dependency is not.

The index is `IndexFlatIP` over 21M passages: ~65 GB, memory-mapped and searched
on CPU. Loading it into GPU memory would take a whole H100 that the policy needs,
and a flat inner-product scan is bandwidth-bound anyway.
"""

import argparse
import asyncio
import json
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchRequest:
    """One HTTP request after validation, without its event-loop future."""

    queries: tuple[str, ...]
    topk: int
    return_scores: bool


@dataclass
class _PendingRequest:
    request: SearchRequest
    future: asyncio.Future


def load_corpus(path):
    """id -> {"title", "text"}. The jsonl is ~21M lines; held as two lists so the
    per-row dict overhead does not triple the footprint."""
    titles, texts = [], []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            # wiki-18 stores the title on the first line of `contents`.
            contents = row.get("contents")
            if contents is not None:
                head, _, body = contents.partition("\n")
                titles.append(head.strip('"'))
                texts.append(body)
            else:
                titles.append(row.get("title", ""))
                texts.append(row.get("text", ""))
    logger.info("corpus: %d passages", len(titles))
    return titles, texts


class Encoder:
    """e5 query encoder. e5 requires the `query: ` prefix and L2-normalised
    output; without either, inner-product scores are not comparable to the
    index's."""

    def __init__(self, path, device="cpu", max_length=256):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).to(device).eval()

    def __call__(self, queries):
        torch = self.torch
        batch = self.tokenizer(
            [f"query: {q}" for q in queries],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            out = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            # e5 pools by mean over unmasked tokens, not by CLS.
            emb = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.cpu().numpy().astype(np.float32)


def search_batch(index, titles, texts, encoder, requests):
    """Run concurrent HTTP requests as one encoder and FAISS operation.

    FAISS and the PyTorch encoder are native, threaded objects. Calling the same
    instances from FastAPI's request thread pool is not safe: the first
    concurrent rollout searches can terminate the whole server. Batching also
    turns N full scans of the 65 GB flat index into one matrix search.
    """
    queries = [query for request in requests for query in request.queries]
    max_topk = max(request.topk for request in requests)
    embeddings = encoder(queries)
    scores, ids = index.search(embeddings, max_topk)

    responses = []
    row_offset = 0
    for request in requests:
        result_groups = []
        for row_scores, row_ids in zip(
            scores[row_offset : row_offset + len(request.queries)],
            ids[row_offset : row_offset + len(request.queries)],
            strict=True,
        ):
            hits = []
            for score, idx in zip(row_scores[: request.topk], row_ids[: request.topk], strict=True):
                idx = int(idx)
                if idx < 0:
                    continue
                doc = {"contents": f'"{titles[idx]}"\n{texts[idx]}'}
                hit = {"document": doc}
                if request.return_scores:
                    hit["score"] = float(score)
                hits.append(hit)
            result_groups.append(hits)
        responses.append({"result": result_groups})
        row_offset += len(request.queries)
    return responses


class RequestBatcher:
    """Serialize native retrieval while coalescing nearby HTTP requests."""

    def __init__(
        self,
        search: Callable[[list[SearchRequest]], list[dict[str, Any]]],
        max_requests: int,
        wait_ms: int,
    ):
        if max_requests < 1:
            raise ValueError(f"max_requests must be positive, got {max_requests}")
        if wait_ms < 0:
            raise ValueError(f"wait_ms must be non-negative, got {wait_ms}")
        self._search = search
        self._max_requests = max_requests
        self._wait_seconds = wait_ms / 1000
        self._queue: asyncio.Queue | None = None
        self._worker: asyncio.Task | None = None
        self._executor: ThreadPoolExecutor | None = None

    async def start(self):
        self._queue = asyncio.Queue()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="search-r1-retrieval")
        self._worker = asyncio.create_task(self._run(), name="search-r1-retrieval-batcher")

    async def stop(self):
        if self._worker is None:
            return
        self._worker.cancel()
        with suppress(asyncio.CancelledError):
            await self._worker
        self._worker = None
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._executor = None

    async def submit(self, request):
        if self._queue is None or self._worker is None or self._worker.done():
            raise RuntimeError("retrieval batch worker is not running")
        future = asyncio.get_running_loop().create_future()
        await self._queue.put(_PendingRequest(request=request, future=future))
        return await future

    async def _collect_batch(self):
        first = await self._queue.get()
        pending = [first]
        deadline = asyncio.get_running_loop().time() + self._wait_seconds
        while len(pending) < self._max_requests:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            try:
                pending.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
            except TimeoutError:
                break
        return pending

    async def _run(self):
        while True:
            pending = await self._collect_batch()
            requests = [item.request for item in pending]
            query_count = sum(len(request.queries) for request in requests)
            logger.info("retrieval batch: %d requests, %d queries", len(requests), query_count)
            try:
                responses = await asyncio.get_running_loop().run_in_executor(self._executor, self._search, requests)
                if len(responses) != len(pending):
                    raise RuntimeError(
                        f"retrieval returned {len(responses)} responses for {len(pending)} HTTP requests"
                    )
            except Exception as error:
                logger.exception("retrieval batch failed")
                for item in pending:
                    if not item.future.done():
                        item.future.set_exception(error)
            else:
                for item, response in zip(pending, responses, strict=True):
                    if not item.future.done():
                        item.future.set_result(response)
            finally:
                for _ in pending:
                    self._queue.task_done()


def build_app(index, titles, texts, encoder, default_topk, batch_max_requests=64, batch_wait_ms=50):
    from fastapi import FastAPI
    from fastapi import HTTPException
    from pydantic import BaseModel

    def run_batch(requests):
        return search_batch(index, titles, texts, encoder, requests)

    batcher = RequestBatcher(run_batch, max_requests=batch_max_requests, wait_ms=batch_wait_ms)

    @asynccontextmanager
    async def lifespan(_app):
        await batcher.start()
        try:
            yield
        finally:
            await batcher.stop()

    app = FastAPI(lifespan=lifespan)

    class Query(BaseModel):
        queries: list[str]
        topk: int | None = None
        return_scores: bool = False

    @app.get("/health")
    def health():
        return {"status": "ok", "passages": len(titles)}

    @app.post("/retrieve")
    async def retrieve(req: Query):
        topk = req.topk if req.topk is not None else default_topk
        if not req.queries or any(not query.strip() for query in req.queries):
            raise HTTPException(status_code=400, detail="queries must contain non-empty strings")
        if topk < 1:
            raise HTTPException(status_code=400, detail="topk must be positive")
        request = SearchRequest(queries=tuple(req.queries), topk=topk, return_scores=req.return_scores)
        return await batcher.submit(request)

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--port", type=int, default=8000)
    # 0.0.0.0, not loopback. Ray actors did not reach a loopback-bound server
    # even co-located on one node, and binding loopback also rules out ever
    # putting the trainer on more than one node. The port is inside the job's
    # container network, not on a public interface.
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--encoder-device", default="cpu")
    ap.add_argument("--faiss-threads", type=int, default=32)
    ap.add_argument("--batch-max-requests", type=int, default=64)
    ap.add_argument("--batch-wait-ms", type=int, default=50)
    args = ap.parse_args()

    if args.batch_max_requests < 1:
        ap.error("--batch-max-requests must be positive")
    if args.batch_wait_ms < 0:
        ap.error("--batch-wait-ms must be non-negative")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    import faiss
    import uvicorn

    faiss.omp_set_num_threads(args.faiss_threads)
    logger.info("loading index %s", args.index)
    # IO_FLAG_MMAP: 65 GB stays on lustre-backed page cache instead of being
    # copied into the process, which would otherwise stall startup for minutes.
    index = faiss.read_index(args.index, faiss.IO_FLAG_MMAP)
    logger.info("index: %d vectors, dim %d", index.ntotal, index.d)

    titles, texts = load_corpus(args.corpus)
    if len(titles) != index.ntotal:
        raise SystemExit(
            f"corpus has {len(titles)} passages but the index has {index.ntotal}; "
            f"they are not the same build and every retrieved id would be wrong"
        )

    encoder = Encoder(args.encoder, device=args.encoder_device)
    logger.info("serving on %s:%d", args.host, args.port)
    app = build_app(
        index,
        titles,
        texts,
        encoder,
        args.topk,
        batch_max_requests=args.batch_max_requests,
        batch_wait_ms=args.batch_wait_ms,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
