import uvicorn
from app import app
from Utils.logger import get_logger
from Utils.Config import get_settings
from fastapi import FastAPI
from controller.chat_controller import router as chat_router
from controller.document_controller import router as doc_router
from controller.evaluation_controller import router as eval_router
from controller.health_controller import router as health_router

logger = get_logger(__name__)
settings = get_settings()

@app.on_event("startup")
async def on_startup():
    logger.info("Starting up AiBuildingAssistanceApi...")
    #init_chromadb()                  # initialise ChromaDB collection
    register_all_routers(app)        # mount all controllers
    logger.info("All routers registered. App ready.")

@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Shutting down gracefully...")

def register_all_routers(app: FastAPI):
    #app.include_router(health_router, prefix="/api/v1", tags=["Health"])
    app.include_router(health_router, prefix="/api/v1", tags=["Health"])
    app.include_router(chat_router, prefix="/api/v1", tags=["Chat"])
    app.include_router(doc_router, prefix="/api/v1", tags=["Documents"])
    #app.include_router(eval_router, prefix="/api/v1", tags=["Evaluation"])


if __name__ == "__main__":
    uvicorn.run(
        "program:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )