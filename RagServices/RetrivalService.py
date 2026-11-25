import torch
import re
import time
import math
from RagServices import QdrantHybrideService as Q
from Model_Class import RetrivalClass
from sentence_transformers import  SentenceTransformer,CrossEncoder
from qdrant_client import QdrantClient
from Utils import  Config,logger
from Model_Class.QuestionModel import QuestionModel
from pydantic import BaseModel

global_variable = Config.get_settings()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEBUG_N = 0  # set by --debug
KS = (1, 3, 5, 10)
_STOP = {"that", "this", "with", "from", "have", "been", "which", "their", "there", "under", "into", "also", "such",
         "these", "those", "shall", "where", "when", "than", "more", "most", "each", "other", "only", "some", "used",
         "using", "apply", "applies", "applicable", "between", "within", "building", "buildings", "code", "ecbc",
         "specifies", "required", "requirement", "requirements", "should", "must", "will", "does", "were", "being"}
POOL: list = []  # top-5 hits of every DB, for pooled labelling (--export-pool)

logger = logger.get_logger(__name__)
class RetrivalService:
    def __init__(self):
        pass

    def retrive(self,question_model,match_with = "answer"):
        logger.info("retrieve start | db_root=%s", global_variable.DB_ROOT)

        client = None  # defined BEFORE the try so `finally` can always check it
        try:
            question = self.build_questions(question_model)

            client = Q.NativeHybridRetriever.make_client(global_variable.DB_ROOT)
            if client is None:
                logger.error("Vector DB client not found | path=%s", global_variable.DB_ROOT)
                raise FileNotFoundError(
                    f"Vector DB client not found for path {global_variable.DB_ROOT}"
                )

            # get_collection() needs a name; list the collections instead
            collections = client.get_collections().collections
            if not collections:
                logger.error("No collections in DB | path=%s", global_variable.DB_ROOT)
                raise FileNotFoundError(f"No collections found in {global_variable.DB_ROOT}")

            collection_name = collections[0].name
            if len(collections) > 1:
                logger.warning("Multiple collections found %s, using %r",
                               [c.name for c in collections], collection_name)

            cfg = {
                "name": collection_name,
                "label": collection_name,
                "embed_model": global_variable.EMBEDDING_MODELS["bge-small-en-v1.5"],
                "query_prefix": "",
            }

            mode = match_with
            reranker = CrossEncoder(global_variable.RERANKER_NAME, max_length=global_variable.RERANK_MAX_LEN, device=DEVICE)


            result = self.run_collection(client, cfg, question, mode, reranker)
            logger.info("retrieve done | collection=%s rows=%d", collection_name, len(result))
            return result

        except Exception:
            logger.exception("retrieve failed | db_root=%s", global_variable.DB_ROOT)
            raise

        finally:
            # Only close if the client was actually created
            if client is not None:
                try:
                    client.close()
                    logger.debug("Qdrant client closed")
                except Exception:
                    # A close error must not mask the original exception
                    logger.warning("Failed to close Qdrant client", exc_info=True)

    def src_key(self,s) -> str:
        return self.norm(s).replace(".pdf", "").replace(" ", "_")
    def parse_gold_pages(self,s) -> set:
        """'ECBC_2024.pdf:62; ECBC_2024.pdf:77-79' -> {('ecbc_2024', 62), ('ecbc_2024', 77), ...}"""
        out = set()
        if s is None or (isinstance(s, float) and math.isnan(s)):
            return out
        for src, a, b in re.findall(r"([^:;,\n]+?)\s*:\s*(\d+)(?:\s*-\s*(\d+))?", str(s)):
            lo, hi = int(a), int(b) if b else int(a)
            out.update((self.src_key(src), p) for p in range(lo, hi + 1))
        return out

    def clause_ids(self,label: str) -> list[str]:
        """'Section 7.2, Table 7.1' -> ['section 7.2', 'table 7.1']  (also keeps bare '7.2')."""
        out = []
        for kind, num in re.findall(r"(section|clause|table|annex|appendix|figure)?\s*([A-Za-z]?\d+(?:\.\d+)*)",
                                    self.norm(label)):
            out.append((f"{kind} {num}").strip())
            out.append(num)
        return list(dict.fromkeys(out))

    def key_numbers(self,answer: str) -> list[str]:
        """Numbers in the ground-truth answer (e.g. '10.8'), the facts a correct chunk must contain."""
        nums = re.findall(r"\d+(?:\.\d+)?", str(answer or ""))
        # ignore years such as 2017 / 2021: they name the standard, they are not the answer
        nums = [n for n in nums if not re.fullmatch(r"(19|20)\d{2}", n)]
        return list(dict.fromkeys(nums))

    def content_words(self,s: str) -> set[str]:
        return {w for w in re.findall(r"[a-z]{4,}", self.norm(s)) if w not in _STOP}
    def build_questions(self,question :QuestionModel) -> list[dict]:
        # ---------- Normalise input to a plain dict ----------
        if isinstance(question, BaseModel):
            data = question.model_dump()  # Pydantic v1: .dict()
        elif isinstance(question, dict):
            data = question
        else:
            raise TypeError(f"Unsupported question type: {type(question)!r}")

        text = (data.get("question") or "").strip()
        if not text:
            logger.error("build_questions: question text missing | data=%r", data)
            raise ValueError("question text is required")

        try:
            answer = str(data.get("answer") or "")

            # ---------- Sources: derive from gold_pages, fall back to source_document ----------
            gold_pages = self.parse_gold_pages(data.get("gold_pages") or "")
            sources = set()
            if data.get("sources"):  # explicit sources, if given
                raw = data["sources"]
                raw = [raw] if isinstance(raw, str) else raw
                sources.update(self.norm(s) for s in raw)
            elif data.get("source_document"):
                sources.add(self.norm(data["source_document"]))

            # ---------- Clauses: JSON field is section_clause ----------
            clauses = self.clause_ids(data.get("section_clause") or data.get("section") or "")
            clauses = list(dict.fromkeys(clauses))  # dedupe, keep order

            entry = {
                "id": data.get("id"),
                "question": text,
                "answer": answer,
                "category": data.get("category"),
                "difficulty": data.get("difficulty"),
                "qtype": data.get("query_type"),  # input query_type -> output qtype
                "sources": sources,
                "clauses": clauses,
                "gold_pages": gold_pages,
                "numbers": self.key_numbers(answer),
                "words": self.content_words(answer),
            }

            logger.debug("build_questions | id=%s sources=%s clauses=%s gold_pages=%s",
                         entry["id"], sources, clauses, gold_pages)
            return entry

        except Exception:
            logger.exception("build_questions failed | question=%r", text[:100])
            raise

    def load_embedder(self,name: str):
        # Guard against a list/None being passed (e.g. EMBEDDING_MODELS is plural)
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Embedding model name must be a non-empty string, got {name!r}")

        # Cache per model so it is not reloaded on every request
        if not hasattr(self, "_embedders"):
            self._embedders = {}
        if name in self._embedders:
            logger.debug("Embedder cache hit | model=%s", name)
            return self._embedders[name]

        logger.info("Loading embedder | model=%s device=%s", name, DEVICE)

        try:
            model = SentenceTransformer(name, device=DEVICE)

            if DEVICE == "cuda":
                # fp16 halves GPU memory (needed to fit BGE-M3 + reranker in 6 GB) and is faster
                model.half()

        except Exception as e:
            # Full traceback, then raise so the caller sees the real cause
            logger.exception("Failed to load embedder | model=%s device=%s", name, DEVICE)
            raise RuntimeError(f"Could not load embedding model {name!r}: {e}") from e

        self._embedders[name] = model
        logger.info("Embedder loaded | model=%s", name)
        return model

    def is_relevant(self,hit: RetrivalClass.Hit, q: dict, mode: str) -> bool:
        """
        clause : chunk is from the right document AND mentions / is labelled with an accepted clause
        answer : chunk contains every key number of the ground-truth answer
        either : clause OR answer   (default; robust to chunking that splits clause headings away)
        """
        if mode == "page":  # right source file AND a gold page
            try:
                page = int(hit.payload.get("page"))
            except (TypeError, ValueError):
                return False
            return (self.src_key(hit.payload.get(global_variable.SOURCE_FIELD, "")), page) in q["gold_pages"]
        text = self.norm(hit.text + " " + hit.payload.get(global_variable.PARENT_TEXT_FIELD, ""))
        section = self.norm(hit.payload.get(global_variable.SECTION_FIELD) or hit.payload.get("parent_title", ""))
        source = self.norm(hit.payload.get(global_variable.SOURCE_FIELD, ""))

        src_ok = (not q["sources"]) or any(
            s and (s in source or source in s or s.split(" ")[0] in source) for s in q["sources"]) \
                 or not source  # payload without a source field: do not penalise

        clause_ok = False
        if src_ok:
            for c in q["clauses"]:
                if "." not in c and " " not in c:  # bare '2' or '4' matches almost anything (e.g. 'CO2'); ignore
                    continue
                if c == section or (section and re.search(rf"(?<![A-Za-z0-9.]){re.escape(c)}(?![0-9]|\.\d)", section)):
                    clause_ok = True
                    break
                if " " in c and c in text:  # 'table 7.1' / 'section 7.2' written in the chunk text
                    clause_ok = True
                    break

        if q["numbers"]:  # numeric answers: every number must appear in the chunk
            answer_ok = all(self.has_number(n, text) for n in q["numbers"])
        else:  # text answers: enough of the ground-truth answer's key words must appear in the chunk
            answer_ok = bool(q["words"]) and len(q["words"] & self.content_words(text)) / len(
                q["words"]) >= global_variable.OVERLAP

        return {"clause": clause_ok, "answer": answer_ok, "either": clause_ok or answer_ok}[mode]

    def src_key(self,s) -> str:
        return self.norm(s).replace(".pdf", "").replace(" ", "_")

    def norm(self,s: str) -> str:
        return re.sub(r"\s+", " ", str(s or "").lower()).strip()

    def has_number(self,n: str, text: str) -> bool:
        """Whole-number match: '5' must not match inside '2.5', '50' or '2015'."""
        return re.search(rf"(?<![\d.]){re.escape(n)}(?![\d]|\.\d)", text) is not None

    def content_words(self,s: str) -> set[str]:
        return {w for w in re.findall(r"[a-z]{4,}", self.norm(s)) if w not in _STOP}

    ##Set the vector name and text field for THIS collection (your 9 DBs may differ). Returns (vector_size, payload_keys, n_points)."""

    def autodetect_fields(self,client: QdrantClient, collection: str):
        logger.info("Inside autodetect_fields, set vector name and text field")
        try:
            global TEXT_FIELD, VECTOR_NAME
            info = client.get_collection(collection)
            vecs = info.config.params.vectors
            logger.info(f"Vector name: {vecs}")
            if isinstance(vecs, dict):  # named vectors
                VECTOR_NAME = next(iter(vecs))
                size = vecs[VECTOR_NAME].size
            else:
                VECTOR_NAME, size = None, vecs.size
            pts, _ = client.scroll(collection, limit=1, with_payload=True, with_vectors=False)
            keys = list(self.flat(pts[0].payload).keys()) if pts else []
            for c in (global_variable.TEXT_FIELD, *global_variable.TEXT_CANDIDATES):
                if c in keys:
                    TEXT_FIELD = c
                    break
            else:
                raise SystemExit(
                    f"[{collection}] no text field found in payload keys {keys}. Set TEXT_FIELD at the top of hybrid_rag.py.")
            return size, keys, info.points_count
        except Exception as e:
            logger.error(f"[{collection}] failed to autodetect field : {e}")
            raise SystemExit(f"[{collection}] failed to autodetect field : {e}")

    def metrics_for(self,ranked_rel: list[bool]) -> dict:
        out = {}
        for k in KS:
            out[f"recall@{k}"] = float(any(ranked_rel[:k]))
        first = next((i for i, r in enumerate(ranked_rel[:10], 1) if r), None)
        out["mrr@10"] = 1.0 / first if first else 0.0
        dcg = sum(1.0 / math.log2(i + 1) for i, r in enumerate(ranked_rel[:10], 1) if r)
        n_rel = sum(ranked_rel[:10])
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_rel + 1))
        out["ndcg@10"] = dcg / idcg if idcg else 0.0
        out["first_rel_rank"] = first or 0
        return out

    def overlap_score(self,hit: RetrivalClass.Hit, q: dict) -> float:
        """Share of the ground-truth answer's key words found in the chunk (shown in --debug to tune --overlap)."""
        return round(len(q["words"] & self.content_words(hit.text)) / len(q["words"]), 2) if q["words"] else -1.0

    def flat(self,payload: dict | None) -> dict:
        """LangChain-style payloads nest section/source under 'metadata'; lift them to the top level."""
        logger.info("Inside flat, pass payload")
        try:
            payload = payload or {}
            meta = payload.get("metadata")
            logger.info(f"Flat meta data: {meta}")
        except Exception as e:
            logger.error(f"Internal system error for flat : {e}")
        return {**meta, **payload} if isinstance(meta, dict) else payload

    def content_words(self,s: str) -> set[str]:
        return {w for w in re.findall(r"[a-z]{4,}", self.norm(s)) if w not in _STOP}

    def run_collection(self,client, cfg, question, mode,  reranker):
        label = cfg.get("label", cfg.get("name"))
        logger.info("run_collection start | collection=%s", label)

        rows = []  # defined before the try so it always exists

        try:
            query = question["question"]
            model = cfg["embed_model"]

            # ---------- Load embedder and validate dimensions ----------
            embedder = self.load_embedder(model)
            size, keys, n = self.autodetect_fields(client, cfg["name"])

            # sentence-transformers renamed this method across versions, so try both
            # Fail clearly if loading returned nothing, instead of a confusing AttributeError
            if embedder is None:
                raise ValueError(f"load_embedder returned None for model {model!r}")

            # Try the new method name first, then the old one
            dim_fn = (getattr(embedder, "get_embedding_dimension", None)
                      or getattr(embedder, "get_sentence_embedding_dimension", None))

            if dim_fn is None:
                # Fallback that works on any version: encode a dummy string and read the shape
                dim = len(embedder.encode("test"))
            else:
                dim = dim_fn()

            if dim != size:
                # ValueError instead of SystemExit: SystemExit is not an Exception
                # subclass, so it would bypass `except Exception` and kill the worker
                raise ValueError(
                    f"[{label}] DB vectors have {size} dims but {model} produces {dim}. "
                    "This DB was built with a different embedding model."
                )
            logger.info("[%s] %d chunks | vector size %d | payload keys: %s",
                        label, n, size, keys)

            # ---------- Build retriever (BM25 + dense + RRF all inside Qdrant) ----------
            ret = Q.NativeHybridRetriever(
                client, cfg["name"], model,
                embedder=embedder, reranker=reranker,
                query_prefix=cfg.get("query_prefix", ""),
            )
            logger.info("[%s] NativeHybridRetriever initialised", label)

            # ---------- Warm-up (keeps model start-up out of latency numbers) ----------
            try:
                ret.search(query, top_k=10)
            except Exception:
                # A warm-up failure is not fatal; the real search below decides
                logger.warning("[%s] warm-up search failed, continuing", label, exc_info=True)

            # ---------- Timed search ----------
            t_start = time.time()
            top, t = ret.search(query, top_k=10)  # dense + BM25 -> RRF -> rerank
            logger.info("[%s] search returned %d hits | total=%.1f ms",
                        label, len(top), t.ms["total"])

            # Helper so the source/section fallback logic lives in one place
            def _section(h):
                return (h.payload.get(global_variable.SECTION_FIELD)
                        or h.payload.get("parent_title"))

            # Save top-5 for later review/pooling
            for h in top[:5]:
                POOL.append((
                    query,
                    str(h.payload.get(global_variable.SOURCE_FIELD)),
                    h.payload.get("page"),
                    _section(h),
                    h.text[:200],
                    label,
                ))

            # ---------- Relevance + metrics ----------
            rel = [self.is_relevant(h, question, mode) for h in top]
            m = self.metrics_for(rel)

            # ---------- Top-3 debug log + review fields ----------
            review = {
                "expected_answer": question["answer"][:300],
                # .get avoids a KeyError if the question has no labelled sources
                "labelled_source": "; ".join(sorted(question.get("sources", []))),
            }
            for i, (h, r) in enumerate(zip(top[:3], rel[:3]), 1):
                logger.debug(
                    "#%d relevant=%s overlap=%s (needs %s) section=%r source=%r | %r",
                    i, r, self.overlap_score(h, question), global_variable.OVERLAP,
                    _section(h), h.payload.get(global_variable.SOURCE_FIELD), h.text[:110],
                )
                review[f"top{i}_relevant"] = r
                review[f"top{i}_source"] = h.payload.get(global_variable.SOURCE_FIELD)
                review[f"top{i}_page"] = h.payload.get("page")
                review[f"top{i}_section"] = _section(h)
                review[f"top{i}_text"] = h.text[:300]

            rows.append({
                "collection": label,
                "question": question["question"],
                "category": question["category"],
                "difficulty": question["difficulty"],
                "qtype": question["qtype"],
                "latency_ms": t.ms["total"],
                "retrieve_ms": t.ms["retrieve_parallel"],
                "rerank_ms": t.ms["rerank"],
                **m,
                **review,
            })

            logger.info("[%s] question done | %.1fs elapsed | relevant_in_top3=%s",
                        label, time.time() - t_start, rel[:3])

        except Exception:
            # Full traceback + context, then propagate so the caller knows it failed
            logger.exception("run_collection failed | collection=%s question_id=%s",
                             label, question.get("id") if isinstance(question, dict) else None)
            raise

        return rows