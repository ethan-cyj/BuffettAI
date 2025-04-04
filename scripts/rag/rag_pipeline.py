import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Any, Optional, Callable
from langchain_core.documents import Document
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.retrieval import main_intialise_retrievers
from rag.prompt_engineering import create_prompt, create_evaluation_prompt
from rag.prompt_engineering import RAG_EVALUATION_SYSTEM_PROMPT, RAGEvaluation
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
        
        if llm_type == "openai" or llm_type == "ollama":
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
        elif self.llm_type == "ollama":
            return self._call_ollama(prompt, context)
        else:
            raise ValueError("Invalid LLM configuration")

    def _call_openai(self, prompt: str, context: str = "") -> str:
        """Internal method for OpenAI API calls"""
        try:
            messages = []
            print("###########PRINTING PROMPT############")
            print(prompt)
            print("###########PRINTING context############")
            print(context)
            context = "You are Warren Buffett, the CEO of Berkshire Hathaway. Use only the docuements provided to answer the question." 
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
    
    def _call_ollama(self, prompt: str, context: str = "") -> str:
        """
        Calls the local Ollama model endpoint.
        Combines prompt and context and sends them to the Ollama API.
        """
        full_prompt = f"{prompt}\n{context}" if context else prompt
        data = {
            "model": "ollama_buffett_v2",
            "prompt": "You are Warren Buffett, the CEO of Berkshire Hathaway. Use only the docuements provided to answer the question. " + full_prompt,
            "stream": False
        }
        try:
            response = requests.post("http://localhost:11434/api/generate", json=data)
            response.raise_for_status()
            json_response = response.json()
            # Extract generated text from the "response" field.
            return json_response.get("response", "")
        except requests.RequestException as e:
            print(f"Error calling Ollama: {e}")
            return f"Error generating response: {str(e)}"


    def generate_report(self, query: str, documents: Optional[List[Document]] = None) -> tuple:
        """
        Generates a report using retrieved documents.
        """
        prompt = create_prompt(query, documents)
        context_text = "\n".join(self.context_memory)
        
        report = self.generate_text(prompt, context=context_text)
        
        # Update context memory
        self.context_memory.append(f"Query: {query}")
        self.context_memory.append(f"Report: {report[:200]}...")
        
        return report

    def evaluate_report(self, query: str, report: str, documents: List[Document]) -> str:
        """
        Evaluates the generated report.
        """
        eval_prompt = create_evaluation_prompt(query, report, documents)
        return self.generate_text(eval_prompt)

    def evaluate_response(self, query: str, response: str, documents: str) -> str:
        """
        Evaluates the generated response.
        """
        response = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": RAG_EVALUATION_SYSTEM_PROMPT},
                {"role": "user", "content": response},
                {"role": "user", "content": f'document={documents}'},
            ],
            response_format=RAGEvaluation,
            temperature=0,
            seed = 42,
            max_tokens=1500
        )
        result = response.choices[0].message.parsed
        if result is None:
            print("Failed to parse response")
            return None
        print(f'Query Relevance: {result.query_relevance}')
        print(f'Data Accuracy: {result.data_accuracy}')
        print(f'Clarity: {result.clarity}')
        print(f'Overall Score: {result.overall_score}')
        return result.overall_score

    def process_query(self, query: str) -> tuple:
        """
        Main method to process a query through the RAG pipeline.
        """
        context_text = "\n".join(self.context_memory)
        documents = main_routing_function(query)
        prompt = create_prompt(query, documents)
        response = self.generate_text(prompt, context=context_text)
        evaluation = self.evaluate_response(prompt, response, documents)
        # report = self.generate_report(query, company_name, documents)
        # evaluation = self.evaluate_report(query, report, documents)
        
        # return report, evaluation
        return {
            "response": response,
            "evaluation": evaluation
        }
    

# testing
# print(os.path.dirname(os.path.abspath(__file__)))
# print(f"Script running from: {os.getcwd()}")
# rag_pipeline = RAGPipeline(llm_type="ollama")
# print(rag_pipeline.process_query("Who are you and tell me about Tesla?", "Tesla"))