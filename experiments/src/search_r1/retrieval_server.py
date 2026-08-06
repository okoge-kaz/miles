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
import json
import logging

import numpy as np

logger = logging.getLogger(__name__)


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


def build_app(index, titles, texts, encoder, default_topk):
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI()

    class Query(BaseModel):
        queries: list[str]
        topk: int | None = None
        return_scores: bool = False

    @app.get("/health")
    def health():
        return {"status": "ok", "passages": len(titles)}

    @app.post("/retrieve")
    def retrieve(req: Query):
        topk = req.topk or default_topk
        emb = encoder(req.queries)
        scores, ids = index.search(emb, topk)
        results = []
        for row_scores, row_ids in zip(scores, ids, strict=True):
            hits = []
            for score, idx in zip(row_scores, row_ids, strict=True):
                if idx < 0:
                    continue
                # The shape local_search_server.py expects, verbatim.
                doc = {"contents": f'"{titles[idx]}"\n{texts[idx]}'}
                hit = {"document": doc}
                if req.return_scores:
                    hit["score"] = float(score)
                hits.append(hit)
            results.append(hits)
        return {"result": results}

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
    args = ap.parse_args()

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
    uvicorn.run(build_app(index, titles, texts, encoder, args.topk), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
