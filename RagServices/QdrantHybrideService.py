"""
Hybrid search done INSIDE Qdrant (no manual BM25 code at query time).

How it works
  * Each chunk is stored with TWO vectors:  "dense" (your embedding)  and  "bm25" (a sparse keyword vector).
  * The sparse vector holds, for every word in the chunk, its BM25 term weight (saturated tf, length-normalised).
    The collection is created with  modifier=IDF , so Qdrant itself multiplies by the word rarity (IDF) at search
    time, using the live statistics of the collection. IDF is always positive in Qdrant.
  * One query_points() call runs dense search + sparse search and fuses them with RRF (k = 60). Metadata filters
    are applied inside both searches. Then the cross-encoder reranks the fused list.

Words are turned into numbers with a stable hash, so nothing has to be saved besides the DB. The tokenizer keeps
clause numbers ("6.5.1") and units ("kW/m2") as single words, which the stock FastEmbed BM25 would split apart.

USE 1: convert your 9 existing DBs (no re-embedding: the stored dense vectors are copied)
    python QdrantHybrideService.py convert --src ./DB --dst ./DB_hybrid
    python QdrantHybrideService.py check   --db ./DB_hybrid/BGE-M3_Embedding/StructureAwareChunking

USE 2: build a NEW DB in the right shape from the start (call this in your ingestion code)
    build_hybrid_db("./DB_new/x", "my_collection", chunks, embed_fn, dense_size=1024)
    # chunks   = [{"id": 0, "text": "...", "payload": {"source": "ECBC_2024.pdf", "page": 7, ...}}, ...]
    # embed_fn = lambda texts: model.encode(texts, normalize_embeddings=True).tolist()

USE 3: evaluate it
    python eval_retrieval.py --test questions.xlsx --db-root ./DB_hybrid --engine native
"""
from __future__ import annotations

import argparse
import os
import re
import time
import zlib
from collections import Counter

from Utils import Config,logger
from qdrant_client import QdrantClient, models
from  Model_Class import RetrivalClass


logger = logger.get_logger(__name__)
global_Variable = Config.get_settings()
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[.\-/][A-Za-z0-9]+)*")  # keeps 6.5.1, kW/m2, ECBC-2017 as one word


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "")]


def token_id(token: str) -> int:
    return zlib.crc32(token.encode("utf-8")) & 0x7FFFFFFF  # stable across runs and machines


# --------------------------------------------------------------------------
# sparse (BM25) encoder
# --------------------------------------------------------------------------
def encode_doc(text: str, avgdl: float) -> models.SparseVector:
    """BM25 term weights of one chunk. Qdrant multiplies these by IDF when searching."""
    tokens = tokenize(text)
    dl = len(tokens)
    idx, val = [], []
    for term, tf in Counter(tokens).items():
        idx.append(token_id(term))
        val.append(tf * (global_Variable.K1 + 1.0) / (tf + global_Variable.K1 * (1.0 - global_Variable.B + global_Variable.B * dl / avgdl)))
    return models.SparseVector(indices=idx, values=val)


def encode_query(text: str) -> models.SparseVector | None:
    # Guard against None / non-string / whitespace-only input
    if not isinstance(text, str) or not text.strip():
        logger.warning("encode_query() got empty or invalid text: %r", text)
        return None

    try:
        # dict.fromkeys removes duplicate terms while preserving order,
        # so each term appears once in the sparse vector
        terms = list(dict.fromkeys(tokenize(text)))

        # Nothing left after tokenization (e.g. only stopwords/punctuation)
        if not terms:
            logger.warning("No tokens produced for query: %r", text[:100])
            return None

        # Map each term to its integer index in the sparse vector space
        indices = [token_id(t) for t in terms]

        logger.debug("encode_query | terms=%d query=%r", len(terms), text[:100])

        return models.SparseVector(
            indices=indices,
            values=[1.0] * len(terms),  # binary weight per unique term
        )

    except Exception:
        # Logs message + full traceback; re-raise so the caller knows encoding failed
        logger.exception("Sparse query encoding failed | query=%r", text[:100])
        raise


# --------------------------------------------------------------------------
# collection set-up and building
# --------------------------------------------------------------------------


def _flat(payload: dict | None) -> dict:
    payload = payload or {}
    meta = payload.get("metadata")
    return {**meta, **payload} if isinstance(meta, dict) else payload


def _text_of(payload: dict | None) -> str:
    p = _flat(payload)
    for k in global_Variable.TEXT_CANDIDATES:
        if k in p:
            return p[k] or ""
    return ""




def discover_dbs(root: str) -> list[str]:
    found = []
    for d, dirs, files in os.walk(root):
        if "meta.json" in files:
            found.append(d)
            dirs[:] = []
    return sorted(found)





def check(db: str, n: int = 20) -> None:
    """Model-free test: use a chunk's own text as the query; the keyword side must bring that chunk back."""
    client = QdrantClient(path=db)
    print(client.get_collections().collections)
    coll = client.get_collections().collections[0].name
    info = client.get_collection(coll)
    print(f"{coll}: {info.points_count} chunks | dense vectors {list(info.config.params.vectors)} | sparse {list(info.config.params.sparse_vectors or {})}")
    pts, _ = client.scroll(coll, limit=min(n, info.points_count), with_payload=True, with_vectors=False)
    ok = 0
    for p in pts:
        q = encode_query(_text_of(p.payload))
        if q is None:
            continue
        top = client.query_points(coll, query=q, using=global_Variable.SPARSE, limit=3).points
        ok += int(p.id in [t.id for t in top])
    print(f"keyword search returns the chunk itself in its top 3 for {ok} of {len(pts)} sampled chunks (expect nearly all)")
    client.close()


# --------------------------------------------------------------------------
# query side: same interface as hybrid_rag.HybridRetriever, so eval_retrieval.py can use it
# --------------------------------------------------------------------------
class NativeHybridRetriever:
    def __init__(self, client, collection, embed_model, reranker_name=None, embedder=None, reranker=None, query_prefix=""):
        self.client, self.collection, self.query_prefix = client, collection, query_prefix
        info = client.get_collection(collection)
        if global_Variable.SPARSE not in (info.config.params.sparse_vectors or {}):
            raise SystemExit(f"[{collection}] has no '{global_Variable.SPARSE}' sparse vector. Convert the DB first: python QdrantHybrideService.py convert ...")
        if embedder is None:
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer(embed_model)
        if reranker is None:
            from sentence_transformers import CrossEncoder
            reranker = CrossEncoder(reranker_name or global_Variable.RERANKER_NAME, max_length=512)
        self.embedder, self.reranker = embedder, reranker

    def retrieve(self, query: str, section=None, source=None):
        try:
            # ---------- Optional payload filters ----------
            conds = []
            if section:
                conds.append(models.FieldCondition(key=global_Variable.SECTION_FIELD,match=models.MatchValue(value=section)))
            if source:
                conds.append(models.FieldCondition(key=global_Variable.SOURCE_FIELD,match=models.MatchValue(value=source)))
            filter_with_conditions = models.Filter(must=conds) if conds else None
            # ---------- Dense query vector ----------
            dense_vec = self.embedder.encode(self.query_prefix + query, normalize_embeddings=True).tolist()
            logger.debug("Dense vector created | query=%r", query[:100])
            prefetch = [models.Prefetch(query=dense_vec,using=global_Variable.DENSE,limit=global_Variable.DENSE_TOP_K,filter=filter_with_conditions)]
            # ---------- Sparse (BM25) query vector ----------
            sparse = encode_query(query)
            if sparse is not None:
                logger.debug("Sparse vector created | query=%r", query[:100])
                prefetch.append(models.Prefetch(query=sparse,using=global_Variable.SPARSE,limit=global_Variable.BM25_TOP_K,filter=filter_with_conditions))
            else:
                logger.info("No sparse terms, using dense-only | query=%r", query[:100])

            # ---------- Fuse with RRF inside Qdrant ----------
            res = self.client.query_points(
                collection_name=self.collection,
                prefetch=prefetch,
                query=models.RrfQuery(rrf=models.Rrf(k=global_Variable.RRF_K)),
                limit=global_Variable.RERANK_POOL,
                with_payload=True,
            ).points

        except Exception:
            # Log full traceback, then re-raise so search() can handle it.
            logger.exception(
                "Hybrid retrieval failed | query=%r section=%s source=%s collection=%s",
                query[:100], section, source, self.collection,
            )
            raise

        logger.info("Retrieved %d candidates | query=%r", len(res), query[:100])

        # Built outside the try so `res` is guaranteed to exist here
        return [
            RetrivalClass.Hit(
                id=str(p.id),
                text=_text_of(p.payload),
                payload=_flat(p.payload),
                rrf_score=float(p.score),
            )
            for p in res
        ]

    def rerank(self, query, hits, top_k):
        if not hits:
            return []
        scores = self.reranker.predict([(query, h.text) for h in hits], batch_size=16, show_progress_bar=False)
        for h, s in zip(hits, scores):
            h.rerank_score = float(s)
        return sorted(hits, key=lambda h: h.rerank_score, reverse=True)[:top_k]

    def search(self, query, section=None, source=None, top_k=None):
        # Validate input early so we fail fast with a clear message
        if not query or not query.strip():
            logger.warning("search() called with an empty query")
            raise ValueError("query must be a non-empty string")

        # Fall back to the global default if top_k is not given
        top_k = top_k or global_Variable.FINAL_TOP_K
        t = RetrivalClass.Timings()

        logger.info(
            "search start | query=%r section=%s source=%s top_k=%d",
            query[:100], section, source, top_k,  # truncate long queries in logs
        )

        total_start = time.perf_counter()

        # ---------- Stage 1: retrieval (embed + dense + sparse + RRF) ----------
        fused = []
        t0 = time.perf_counter()
        try:
            fused = self.retrieve(query, section, source)
        except Exception:
            # logger.exception logs the message AND the full traceback
            logger.exception(
                "Retrieval failed | query=%r section=%s source=%s",
                query[:100], section, source,
            )
            raise  # no candidates -> cannot continue, let the caller handle it
        finally:
            # Record timing even when the stage fails
            t.ms["retrieve_parallel"] = (time.perf_counter() - t0) * 1000

        # RRF happens inside retrieve(), so it is tracked as 0 here
        # (kept so the Timings keys stay consistent for downstream consumers)
        t.ms["rrf"] = 0.0

        logger.debug("retrieve returned %d candidates in %.1f ms",
                     len(fused), t.ms["retrieve_parallel"])

        # Nothing retrieved: skip reranking and return an empty result
        if not fused:
            logger.warning("No candidates found | query=%r", query[:100])
            t.ms["rerank"] = 0.0
            t.ms["total"] = (time.perf_counter() - total_start) * 1000
            return [], t

        # ---------- Stage 2: reranking ----------
        t0 = time.perf_counter()
        try:
            top = self.rerank(query, fused, top_k)
        except Exception:
            # Degrade gracefully: the fused order from RRF is still a decent ranking
            logger.exception(
                "Rerank failed, falling back to fused order | query=%r", query[:100]
            )
            top = fused[:top_k]
        finally:
            t.ms["rerank"] = (time.perf_counter() - t0) * 1000

        # Sum of stages (same as before)
        t.ms["total"] = t.ms["retrieve_parallel"] + t.ms["rrf"] + t.ms["rerank"]

        logger.info(
            "search done | results=%d retrieve=%.1fms rerank=%.1fms total=%.1fms",
            len(top), t.ms["retrieve_parallel"], t.ms["rerank"], t.ms["total"],
        )
        return top, t

    def make_client(self,path: str = global_Variable.DB_ROOT) -> QdrantClient:
        logger.info(
            "Inside make_client, Open the local Qdrant folder (embedded mode, no server). Fails if the folder does not exist.")
        if not os.path.isdir(path):
            logger.error(f"Quadrant DB not present at {path}")
            raise SystemExit(
                f"Qdrant folder not found: {path}\nPass the folder you used with QdrantClient(path=...) via --path.")
        try:
            client = QdrantClient(path=path)
            names = [c.name for c in client.get_collections().collections]
            logger.info(f"{len(names)} DB found")
        except Exception as e:
            # local mode allows one process at a time: close notebooks / other scripts that have the folder open
            logger.error(f"Internal system error for opening DB : {e}")
            raise SystemExit(
                f"Could not open {path}: {e}\nIf another notebook or script has this folder open, close it first.")
        print(f"Opened {path}; collections: {names}")
        return client

# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert", help="copy every DB under --src into a hybrid DB under --dst (no re-embedding)")
    c.add_argument("--src", required=True)
    c.add_argument("--dst", required=True)
    c.add_argument("--only", default=None, help="convert only DBs whose path contains this text")
    k = sub.add_parser("check", help="verify the sparse side of one converted DB (needs no model)")
    k.add_argument("--db", required=True)
    a = ap.parse_args()

    check(a.db)
