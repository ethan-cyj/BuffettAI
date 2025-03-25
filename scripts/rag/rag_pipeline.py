import os
from dotenv import load_dotenv
import openai 
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

# call custom LLM
def call_custom_llm(prompt: str, context: str = "") -> str:
    """
    Placeholder function to call your custom LLM (@ethan, @yucai)
    The function should take a prompt and optional context, then return a generated text.
    In production, replace this with the actual API call or function to your LLM.
    """
    # For demonstration, we simply return the prompt combined with context.
    # Replace this with your actual generation call.
    combined_input = context + "\n" + prompt if context else prompt
    return "Custom LLM Response based on:\n" + combined_input

class RAGPipeline:
    def __init__(self, raw_documents):
        """
        Initializes the pipeline:
         - Splits raw documents into text chunks.
         - Builds BM25 and FAISS retrievers.
         - Combines them into an ensemble retriever.
         - Initializes a context memory for feedback loop.
        Expects raw_documents as a list of dicts with at least a 'content' field.
        """
        # Split documents into chunks.
        self.documents = split_documents(raw_documents)
        
        # Build BM25 retriever.
        corpus = [doc.page_content for doc in self.documents]  

        bm25_model = build_bm25(corpus)
        
        self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        self.bm25_retriever.k = 5
        
        # Build FAISS retriever.
        faiss_vectorstore = build_faiss_index(self.documents)
        self.faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Create an ensemble retriever.
        self.ensemble_retriever = create_ensemble_retriever(self.bm25_retriever, self.faiss_retriever)
        
        # Initialize context memory for feedback loop (stores conversation history).
        self.context_memory = []

    def retrieve_and_rerank(self, query: str, retriever, top_k: int = 5):
        """
        Retrieves documents using the provided retriever and then reranks them.
        """
        docs = retriever.get_relevant_documents(query)
        return rerank_documents(query, docs, top_k=top_k)
    
    def generate_report(self, query: str, company_name: str, retriever):
        """
        Retrieves documents, creates a prompt, adds context from previous interactions,
        and uses the custom LLM to generate a report.
        """
        # Retrieve and rerank documents.
        retrieved_docs = self.retrieve_and_rerank(query, retriever)
        
        # Create the base prompt using the retrieved documents.
        prompt = create_prompt(company_name, retrieved_docs)
        
        # Incorporate feedback context (if any) into the prompt.
        # Here we simply concatenate the context memory into a single string.
        context_text = "\n".join(self.context_memory)
        
        # Call your custom LLM with the prompt and context.
        report = call_custom_llm(prompt, context=context_text)
        
        # Update context memory: you can choose to store just the query/response pair
        # or additional details. Here, we store the generated report.
        self.context_memory.append(f"User Query: {query}")
        self.context_memory.append(f"LLM Report: {report}")
        
        return report, retrieved_docs

    def evaluate_report(self, query: str, report: str, retrieved_docs):
        """
        Uses the custom LLM (or another LLM) to evaluate the generated report.
        This function calls the evaluation prompt and returns the evaluation.
        """
        eval_prompt = create_evaluation_prompt(query, report, retrieved_docs)
        # You could also use your custom LLM here, but for now, we call the placeholder.
        evaluation = call_custom_llm(eval_prompt)
        return evaluation

    def process_query(self, query: str, company_name: str, method: str = "ensemble"):
        """
        Processes the query using one of the retrieval methods:
          - "bm25": BM25 only.
          - "faiss": FAISS only.
          - "ensemble": Ensemble (BM25 + FAISS).
        Returns the generated report and its evaluation.
        """
        if method == "bm25":
            retriever = self.bm25_retriever
        elif method == "faiss":
            retriever = self.faiss_retriever
        else:
            retriever = self.ensemble_retriever
        
        report, retrieved_docs = self.generate_report(query, company_name, retriever)
        evaluation = self.evaluate_report(query, report, retrieved_docs)
        return report, evaluation
