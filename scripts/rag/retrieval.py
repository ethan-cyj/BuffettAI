import numpy as np
from rank_bm25 import BM25Okapi
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# Compare 2 different retrieval methods and evaluate

def build_bm25(corpus):
    """
    Builds a BM25 model from a list of texts (corpus).
    """
    return BM25Okapi(corpus)

def build_faiss_index(documents):
    """
    Uses OpenAI embeddings to build a FAISS vectorstore from documents.
    """
    embeddings = OpenAIEmbeddings()
    # FAISS vectorstore will build the index based on document 'content'
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

def create_ensemble_retriever(bm25_retriever, faiss_retriever):
    """
    Combines BM25 and FAISS retrievers into an ensemble retriever.
    """
    return EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever])
