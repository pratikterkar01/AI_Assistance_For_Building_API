from datasets import Dataset
from Services.rag_service import ask_question
from Utils.logger import get_logger
from ragas import experiment
from ragas.metrics.collections import Faithfulness, AnswerRelevancy ,ContextRecall
from pydantic import BaseModel

logger = get_logger(__name__)

async def run_evaluation(questions: list, ground_truths: list) -> dict:
    data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for q, gt in zip(questions, ground_truths):
        result = ask_question(q)
        data["question"].append(q)
        data["answer"].append(result["answer"])
        data["contexts"].append([s["content"] for s in result["sources"]])
        data["ground_truth"].append(gt)

    dataset = Dataset.from_dict(data)
    #scores = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_recall])
    scores = await run_evaluation(dataset)
    df = scores.to_pandas()

    return {
        "faithfulness": round(df["faithfulness"].mean(), 4),
        "answer_relevancy": round(df["answer_relevancy"].mean(), 4),
        "context_recall": round(df["context_recall"].mean(), 4),
    }

# Define experiment result structure
class ExperimentResult(BaseModel):
    faithfulness: float
    answer_relevancy: float
# Create experiment function
@experiment(ExperimentResult)
async def run_evaluation(row):
    faithfulness = Faithfulness()
    answer_relevancy = AnswerRelevancy()
    context_recall = ContextRecall()
    faith_result = await faithfulness.ascore(
        response=row.response,
        retrieved_contexts=row.contexts
    )

    relevancy_result = await answer_relevancy.ascore(
        user_input=row.user_input,
        response=row.response
    )

    context_recall_result = await context_recall.ascore(
        user_input=row.user_input,
        response = row.response
    )

    return ExperimentResult(
        faithfulness=faith_result.value,
        answer_relevancy=relevancy_result.value,
        context_recall = context_recall_result.value
    )