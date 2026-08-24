from fastapi import APIRouter, UploadFile, File, Form
from Model_Class.IngestResponse import IngestResponse
from Services.IngestionService import load_Documents, split_Documents
from Services.vectorstore_service import add_documents
from Utils.logger import get_logger
import shutil, os

router = APIRouter()
logger = get_logger(__name__)


@router.post("/documents/upload", response_model=IngestResponse)
async def upload_document(
        file: UploadFile = File(...),
        doc_type: str = Form("ashrae")
):
    save_path = f"./data/raw/{file.filename}"
    os.makedirs("./data/raw", exist_ok=True)

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    docs = load_Documents(save_path, doc_type)
    chunks = split_Documents(docs)
    count = add_documents(chunks)

    return IngestResponse(
        status="success",
        chunks_added=count,
        collection="building_docs"
    )