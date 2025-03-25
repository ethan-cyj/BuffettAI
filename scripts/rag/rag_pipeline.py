import os
from dotenv import load_dotenv
import openai
from typing import List, Dict, Any, Optional, Callable
from langchain_core.documents import Document
from rag.text_splitter import split_documents
from rag.retrieval import build_bm25, build_faiss_index, create_ensemble_retriever
from rag.reranker import rerank_documents
from rag.prompt_engineering import create_prompt, create_evaluation_prompt

from langchain_community.retrievers import BM25Retriever

# load env variables from .env
load_dotenv()

# set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("API key is not set")

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class RAGPipeline:
    def __init__(
        self, 
        raw_documents,
        llm_type: str = "openai",  # "openai" or "custom"
        custom_llm: Optional[Callable] = None,
        openai_model: str = "gpt-3.5-turbo",
        openai_client: Optional[openai.OpenAI] = None
    ):
        """
        Initialize RAG pipeline with LLM options.
        
        Args:
            raw_documents: Input documents for retrieval
            llm_type: "openai" or "custom"
            custom_llm: Function(prompt: str, context: str) -> str
            openai_model: OpenAI model name if using OpenAI
        """
        self.documents = split_documents(raw_documents)
        self.context_memory = []
        self.llm_type = llm_type
        self.custom_llm = custom_llm
        self.openai_model = openai_model
        
        if llm_type == "openai":
            self.client = openai_client or openai.OpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
            if not self.client.api_key:
                raise ValueError("OPENAI_API_KEY not found in .env file")
        
        self._initialize_retrievers()

    def _initialize_retrievers(self):
        """Initialize BM25 and FAISS retrievers"""
        # BM25 Retriever
        self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        self.bm25_retriever.k = 5
        
        # FAISS Retriever
        self.faiss_retriever = build_faiss_index(self.documents).as_retriever(search_kwargs={"k": 5})
        
        # Ensemble Retriever
        self.ensemble_retriever = create_ensemble_retriever(
            self.bm25_retriever, 
            self.faiss_retriever
        )
    
    def generate_text(self, prompt: str, context: str = "") -> str:
        """
        Unified text generation interface.
        Routes to OpenAI or custom LLM based on configuration.
        """
        if self.llm_type == "custom" and self.custom_llm:
            return self.custom_llm(prompt, context)
        elif self.llm_type == "openai":
            return self._call_openai(prompt, context)
        else:
            raise ValueError("Invalid LLM configuration")

    def _call_openai(self, prompt: str, context: str = "") -> str:
        """Internal method for OpenAI API calls"""
        try:
            import openai
            messages = []
            if context:
                messages.append({"role": "system", "content": context})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                temperature=0.7,
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI: {e}")
            return f"Error generating response: {str(e)}"

    def retrieve_and_rerank(self, query: str, retriever, top_k: int = 5) -> List[Document]:
        """
        Retrieves and reranks documents.
        """
        docs = retriever.get_relevant_documents(query)
        return rerank_documents(query, docs, top_k=top_k)

    def generate_report(self, query: str, company_name: str, retriever) -> tuple:
        """
        Generates a report using retrieved documents.
        """
        retrieved_docs = self.retrieve_and_rerank(query, retriever)
        prompt = create_prompt(query, company_name, retrieved_docs)
        context_text = "\n".join(self.context_memory)
        
        report = self.generate_text(prompt, context=context_text)
        
        # Update context memory
        self.context_memory.append(f"Query: {query}")
        self.context_memory.append(f"Report: {report[:200]}...")
        
        return report, retrieved_docs

    def evaluate_report(self, query: str, report: str, retrieved_docs: List[Document]) -> str:
        """
        Evaluates the generated report.
        """
        eval_prompt = create_evaluation_prompt(query, report, retrieved_docs)
        return self.generate_text(eval_prompt)

    def process_query(self, query: str, company_name: str, method: str = "ensemble") -> tuple:
        """
        Main method to process a query through the RAG pipeline.
        """
        retriever = {
            "bm25": self.bm25_retriever,
            "faiss": self.faiss_retriever,
            "ensemble": self.ensemble_retriever
        }.get(method, self.ensemble_retriever)
        
        report, retrieved_docs = self.generate_report(query, company_name, retriever)
        evaluation = self.evaluate_report(query, report, retrieved_docs)
        
        return report, evaluation