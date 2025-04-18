import os
import logging
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag.rag_pipeline import RAGPipeline  # Your existing RAG pipeline
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

origins = [
    "http://localhost:5173",
    "http://localhost:5174"
]

# Add CORS middleware to your app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        
    allow_credentials=True,
    allow_methods=["*"],          
    allow_headers=["*"],          
)

@app.on_event("startup")
async def startup_event():
    logger.info(f"Server starting in: {os.getcwd()}")
    logger.info(f"Files in data directory: {os.listdir('data') if os.path.exists('data') else 'No data dir'}")
    logger.info("Initializing RAG pipeline...")
    global rag_pipeline
    # rag_pipeline = RAGPipeline(llm_type="ollama")
    rag_pipeline = RAGPipeline(llm_type="openai")
    logger.info("RAG pipeline initialized")
# init rag pipeline
# rag_pipeline = RAGPipeline(llm_type="ollama")
rag_pipeline = RAGPipeline(llm_type="openai")

# req model
class QueryRequest(BaseModel):
    query: str


# query endpoint
@app.post("/query")
async def process_query(request: QueryRequest):
    if not request.query:
        raise HTTPException(status_code=400, detail="Missing query field")

    try:
        # process query with rag pipeline
        response = rag_pipeline.process_query(request.query)
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Error processing query:\n{tb}")

    # return {"report": report, "evaluation": evaluation}
    return response

# root endpoint, test
@app.get("/")
async def root():
    return {"message": "RAG backend is running."}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
