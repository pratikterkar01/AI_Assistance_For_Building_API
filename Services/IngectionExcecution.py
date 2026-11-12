import shutil, os
import glob
import hashlib
import re
import json
from Utils import Config,logger
from Services import StructureAwareChunking
from Services import EmbedingService as Embeding
from langchain_core.documents import Document
from Services import IngestionService as Ingestion
from fastapi import  UploadFile, File
from pathlib import Path

global_Variable = Config.get_settings()

dbRoot = global_Variable.DB_ROOT
logger = logger.get_logger(__name__)
class IngectionExccution:
    def __init__(self):
        pass

    # ============================== small helpers ===============================

    ##
    def doc_slug(self,document_name):
        base = os.path.splitext(os.path.basename(str(document_name)))[0]
        base = re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_")[:25] or "doc"
        return f"{base}_{hashlib.md5(str(document_name).encode('utf-8')).hexdigest()[:6]}"


    def chunks_fingerprint(self,chunks):
        h = hashlib.md5()
        for c in chunks:
            h.update(c.page_content.encode("utf-8"))
        return h.hexdigest()


    def load_manifest(self,db_path):
        path = os.path.join(db_path, global_Variable.MANIFEST_FILE)
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            logger.error(f"Manifest {path} is corrupt; backing it up and starting fresh")
            os.replace(path, path + ".corrupt")
            return {}


    def save_manifest(self,db_path, manifest):
        logger.info("Inside save_manifest ")
        os.makedirs(db_path, exist_ok=True)
        tmp = os.path.join(db_path, global_Variable.MANIFEST_FILE + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        os.replace(tmp, os.path.join(db_path, global_Variable.MANIFEST_FILE))


    def save_parents(self,db_path, parents, source):
        """Merge this PDF's parents into parents.json (other PDFs are kept)."""
        path = os.path.join(db_path, global_Variable.PARENTS_FILE)
        data = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        data = {pid: d for pid, d in data.items() if d["metadata"].get("source") != source}
        for p in parents:
            data[p.metadata["parent_id"]] = {"page_content": p.page_content,
                                             "metadata": p.metadata}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)


    def load_parents(self,db_path):
        with open(os.path.join(db_path, global_Variable.PARENTS_FILE), encoding="utf-8") as f:
            data = json.load(f)
        return {pid: Document(**d) for pid, d in data.items()}


    def parents_present(self,db_path, source):
        try:
            with open(os.path.join(db_path, global_Variable.PARENTS_FILE), encoding="utf-8") as f:
                return any(d["metadata"].get("source") == source for d in json.load(f).values())
        except (FileNotFoundError, json.JSONDecodeError):
            return False


    # ============================ the main function =============================
    def execute_all_combinations(self,documen_parse, document_name):
        logger.info(f"Inside execute_all_combinations ")
        embedding_service = Embeding.EmbedingService()
        db_root = global_Variable.DB_ROOT
        logger.info(f"\n##### Document: {document_name}")

        chunk_factories = {
            "StructureAwareChunking": lambda:
            StructureAwareChunking.StructureAwareChunking()
            .structure_aware_chunking(documen_parse),
        }
        chunk_cache = {}

        def get_chunks(name):
            if name not in chunk_cache:
                logger.info(f"Generating chunks: {name}")
                chunks = chunk_factories[name]()
                for c in chunks:
                    c.metadata["source"] = document_name
                chunk_cache[name] = chunks
            return chunk_cache[name]

        embedding_models = global_Variable.EMBEDDING_MODELS

        results = {"done": [], "skipped": [], "failed": {},"collection_name":"","chunk_count":0}
        vector_stores = {}

        for embedding_folder, model_name in embedding_models.items():
            embeddings = None  # load the model only if something needs building

            for chunking_name in chunk_factories:
                combination_name = f"{embedding_folder}_{chunking_name}"
                collection_name = combination_name.replace("-", "_")
                results["collection_name"] = collection_name
                db_path = os.path.join(db_root, f"{embedding_folder}_Embedding", chunking_name)
                is_hier = chunking_name == "HierarchicalChunking"
                stage = "chunk"
                try:
                    chunks = get_chunks(chunking_name)
                    fingerprint = self.chunks_fingerprint(chunks)
                    results["chunk_count"] += len(chunks)
                    # ---- resume check: only for THIS pdf inside this DB ----
                    stage = "check"

                    entry = self.load_manifest(db_path).get(document_name)
                    if (
                            entry
                            and entry.get("fingerprint") == fingerprint
                            and entry.get("count") == len(chunks)
                            and embedding_service.count_source(
                        db_path, collection_name, document_name) == len(chunks)
                            and (not is_hier or  self.parents_present(db_path, document_name))
                    ):
                        logger.info(f"Skipping (already complete): {combination_name}")
                        results["skipped"].append(combination_name)
                        continue
                    stage = "embed"

                    if embeddings is None:
                        logger.info(f"\nLoading embedding: {embedding_folder}")
                        embeddings = embedding_service.embed(model_name=model_name)

                    logger.info(f"\nCreating: {combination_name}  ({len(chunks)} chunks)")
                    logger.info(f"Database path: {db_path}")

                    # forget the old manifest entry of THIS pdf; vectore_store()
                    # removes its old vectors and adds the new ones
                    stage = "store"
                    manifest = self.load_manifest(db_path)
                    manifest.pop(document_name, None)
                    self.save_manifest(db_path, manifest)

                    vector_store = embedding_service.vectore_store(
                        embeddings=embeddings,
                        langchain_chunks=chunks,
                        collection_name=collection_name,
                        db_path=db_path,
                        source=document_name,
                    )


                    # manifest entry LAST = "this pdf is fully in the DB"
                    manifest = self.load_manifest(db_path)
                    manifest[document_name] = {"fingerprint": fingerprint, "count": len(chunks)}
                    self.save_manifest(db_path, manifest)

                    vector_stores[combination_name] = vector_store
                    results["done"].append(combination_name)
                    logger.debug(f"Completed: {combination_name}")

                except Exception as e:
                    # Developer: full traceback with stage and combination
                    logger.exception(
                        f"FAILED [{stage}] {combination_name}: {type(e).__name__}: {e}")

                    # User: short, safe message (no paths, no stack traces)
                    results["failed"][combination_name] = {
                        "stage": stage,
                        "message": self.friendly_message(stage, e),
                        "error_type": type(e).__name__,
                    }

                    # remove only this pdf's partial vectors, keep other PDFs
                    try:
                        embedding_service.delete_source(db_path, collection_name, document_name)
                    except Exception:
                        logger.exception(
                            f"Cleanup failed for {document_name} in {combination_name}; "
                            "partial vectors may remain")

                    # drop the manifest entry so the next run retries this document
                    try:
                        manifest = self.load_manifest(db_path)
                        if manifest.pop(document_name, None) is not None:
                            self.save_manifest(db_path, manifest)
                    except Exception:
                        logger.exception(f"Could not update manifest for {document_name}")

                    embedding_service.clear_cuda()

            embeddings = None
            embedding_service.clear_cuda()

        logger.debug(f"\n=== Summary for {document_name} ===")
        logger.debug(f"Built:   {len(results['done'])}")
        logger.debug(f"Skipped: {len(results['skipped'])}")
        logger.debug(f"Failed:  {len(results['failed'])}")
        for name, err in results["failed"].items():
            logger.debug(f"  - {name}: {err}")
        if results["failed"]:
            logger.debug("Rerun the script to retry only the failed combinations.")

        return  results



    def collect_pdfs(self,source):
        try:
            """Accept a folder, a single PDF path, or a list of PDF paths."""
            if isinstance(source, (list, tuple)):
                pdfs = list(source)
            elif os.path.isdir(source):
                pdfs = sorted(glob.glob(os.path.join(source, "*.pdf")))
            else:
                pdfs = [source]
            missing = [p for p in pdfs if not os.path.isfile(p)]
            if missing:
                raise FileNotFoundError(f"PDF not found: {missing}")
            if not pdfs:
                raise FileNotFoundError(f"No PDFs found in: {source}")
            return pdfs
        except Exception as e:
            logger.error("Error in collect_pdfs: %s", e)
            return []


    def run_experiment(self):
        final_result = []
        txt_files = sorted(Path(global_Variable.PARSED_DIR).glob("*.txt"))

        for path in txt_files:
            documen_parse = Ingestion.Ingestion.load_parsed_document(path)
            document_name = f"{path.stem}.pdf"  # same name as the original PDF

            result =   self.execute_all_combinations(documen_parse, document_name)
            final_result.append(result)
        logger.debug(f"Final result: {final_result}")
        return final_result

    def parse_and_save_all_pdfs(self,file: UploadFile = File(...)):
        try:
            logger.info("Inside parse_and_save_all_pdfs for parsing and save parse in text")
            save_path = f"./data/raw/{file.filename}"
            os.makedirs("./data/raw", exist_ok=True)

            with open(save_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            logger.info(f"Saved parsed PDF to ./data/raw/{file.filename}")
            pdf_Names = self.collect_pdfs(global_Variable.UPLOAD_FILE_FOLDER)
            logger.info(f"pdf_Names: {pdf_Names}")
            failed_documents = []
            for pdf in pdf_Names:
                text_path = Ingestion.Ingestion.save_parsed_document(pdf)  # skips if JSON already exists
                if text_path == "":
                    failed_documents.append(pdf)



            logger.info(f"Done. Parsed {len(pdf_Names)} PDFs. failed paths: {failed_documents}")
            return f"failed parsing path {failed_documents}"
        except Exception as e:
            logger.error(f"Error while parsing and save pdf as txt file {e}")
            return f"Error while parsing and save pdf as txt file"

    def friendly_message(stage, exc):
        if isinstance(exc, FileNotFoundError):
            return "The file was not found."
        if isinstance(exc, MemoryError) or "out of memory" in str(exc).lower():
            return "The server ran out of memory processing this document. Try a smaller file."
        return global_Variable.STAGE_MESSAGES.get(stage, "An unexpected error occurred while processing this document.")

'''if __name__ == "__main__":
    #parse_and_save_all_pdfs()
     pdf_names = run_experiment()
    #run_experiment()'''


