from fastapi import APIRouter, UploadFile, File, Form
from Model_Class.IngestResponse import IngestResponse
from Services.IngectionExcecution import IngectionExccution
#from Services.vectorstore_service import add_documents
from Utils.logger import get_logger


router = APIRouter()
logger = get_logger(__name__)


@router.post("/documents/upload", response_model=IngestResponse)
async def upload_document( file: UploadFile = File(...), doc_type: str = Form("ashrae")
):

    ## here we have copied then start the parsing to the docline doc and save the text file of it
    ingection_excecution = IngectionExccution()
    #ingection_excecution.parse_and_save_all_pdfs(file)
    ## create the vector DB
    result = ingection_excecution.run_experiment()
    '''count = add_documents(chunks)'''

    return IngestResponse(
        status="success",
        final_vector_db_result = result
    )