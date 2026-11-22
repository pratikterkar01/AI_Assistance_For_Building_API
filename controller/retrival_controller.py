from fastapi import APIRouter, UploadFile, File, Form
from Model_Class.QuestionModel import QuestionModel
from Services.RetrivalService import RetrivalService
from Utils.logger import get_logger
import json

router = APIRouter()
logger = get_logger(__name__)


@router.post("/retrival")
async def retrival_endpoint(item: QuestionModel):
    ## here we have copied then start the parsing to the docline doc and save the text file of it
    retrival = RetrivalService()
    question = dict(item)
    retival_chunks = retrival.retrive(question)
    return {
        "question": item.question,
        "retrival": retival_chunks,
    }
