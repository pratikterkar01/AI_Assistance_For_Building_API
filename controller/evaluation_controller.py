from fastapi import APIRouter
from Model_Class.EvalRequest import EvalRequest
from Model_Class.EvalResponse import EvalResponse
from Services.evaluation_service import run_evaluation

router = APIRouter()

@router.post("/evaluate", response_model=EvalResponse)
async def evaluate(request: EvalRequest):
    scores = run_evaluation(request.questions, request.ground_truths)
    return EvalResponse(**scores)