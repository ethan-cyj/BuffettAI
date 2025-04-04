import os
import logging
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag.rag_pipeline import RAGPipeline  # Your existing RAG pipeline
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    logger.info(f"Server starting in: {os.getcwd()}")
    logger.info(f"Files in data directory: {os.listdir('data') if os.path.exists('data') else 'No data dir'}")
    logger.info("Initializing RAG pipeline...")
    global rag_pipeline
    rag_pipeline = RAGPipeline(llm_type="ollama")
    logger.info("RAG pipeline initialized")
# init rag pipeline
rag_pipeline = RAGPipeline(llm_type="ollama")

# req model
class QueryRequest(BaseModel):
    query: str
    company: str

# query endpoint
@app.post("/query")
async def process_query(request: QueryRequest):
    if not request.query or not request.company:
        raise HTTPException(status_code=400, detail="Missing query or company field")

    try:
        # process query with rag pipeline
        # report, evaluation = rag_pipeline.process_query(request.query, request.company)
        response = rag_pipeline.process_query(request.query, request.company)
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Error processing query:\n{tb}")

    # return {"report": report, "evaluation": evaluation}
    return {"response": response}

# root endpoint, test
@app.get("/")
async def root():
    return {"message": "RAG backend is running."}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
