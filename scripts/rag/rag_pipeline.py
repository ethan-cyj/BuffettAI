import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Any, Optional, Callable
from langchain_core.documents import Document

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.retrieval import main_intialise_retrievers
from rag.prompt_engineering import create_prompt, create_evaluation_prompt
from rag.agent import main_routing_function


# load env variables from .env
load_dotenv()

class RAGPipeline:
    def __init__(
        self, 
        llm_type: str = "openai",  # "openai" or "custom"
        custom_llm: Optional[Callable] = None,
    ):
        """
        Initialize RAG pipeline with LLM options.
        
        Args:
            llm_type: "openai" or "custom"
            custom_llm: Function(prompt: str, context: str) -> str
        """
        self.context_memory = []
        self.llm_type = llm_type
        self.custom_llm = custom_llm
        
        if llm_type == "openai":
            self.client = OpenAI()
        
        self._initialize_retrievers()

    def _initialize_retrievers(self):
        """Initialize BM25 and FAISS retrievers"""
        bm25_index_path = "bm25_indexes"
        faiss_index_path = "faiss_indexes"
        bm25_exists = os.path.exists(bm25_index_path) and os.listdir(bm25_index_path)
        faiss_exists = os.path.exists(faiss_index_path) and os.listdir(faiss_index_path)
        if bm25_exists and faiss_exists:
            print("BM25 and FAISS indexes found. Skipping initialization.")
        else:
            print("Initializing BM25 and FAISS indexes...")
            main_intialise_retrievers()
    
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
            messages = []
            if context:
                messages.append({"role": "system", "content": context})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7,
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI: {e}")
            return f"Error generating response: {str(e)}"

    def generate_report(self, query: str, company_name: str, documents: Optional[List[Document]] = None) -> tuple:
        """
        Generates a report using retrieved documents.
        """
        prompt = create_prompt(query, company_name, documents)
        context_text = "\n".join(self.context_memory)
        
        report = self.generate_text(prompt, context=context_text)
        
        # Update context memory
        self.context_memory.append(f"Query: {query}")
        self.context_memory.append(f"Report: {report[:200]}...")
        
        return report

    def evaluate_report(self, query: str, report: str, documents: List[Document]) -> str:
        """
        Evaluates the generated report.
        Evaluates the generated report.
        """
        eval_prompt = create_evaluation_prompt(query, report, documents)
        return self.generate_text(eval_prompt)

    def process_query(self, query: str, company_name: str) -> tuple:
        """
        Main method to process a query through the RAG pipeline.
        """
        documents = main_routing_function(query)
        report = self.generate_report(query, company_name, documents)
        evaluation = self.evaluate_report(query, report, documents)
        
        return report, evaluation
    

# testing
print(os.path.dirname(os.path.abspath(__file__)))
print(f"Script running from: {os.getcwd()}")
rag_pipeline = RAGPipeline()
print(rag_pipeline.process_query("What did Buffett say about Tesla?", "Tesla"))