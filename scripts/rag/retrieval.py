import sys
import os
import numpy as np
import pickle
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain.embeddings import SentenceTransformerEmbeddings
from typing import List
from langchain.schema import Document
from reranker import rerank_documents
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings

from text_splitter import chunk_trades, chunk_qna_data, RecursiveCharacterTextSplitter
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import load_news_pickle, load_shareholder_letters_pickle

load_dotenv()
EXCEL_DATA_SOURCES = ["brka_trades.xlsx", "buffet_qna.xlsx"]


def build_bm25(corpus):
    """
    Builds a BM25 model from a list of texts (corpus).
    """
    return BM25Okapi(corpus)

def get_embeddings(embedding_type="openai"):
    """
    Get consistent embeddings model.
    """
    if embedding_type == "sentence_transformers":
        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"  # or whatever model you used for Excel
        )
    else:
        return OpenAIEmbeddings()

def build_faiss_index(documents, save_path = "faiss_index"):
    """
    Uses OpenAI embeddings to build a FAISS vectorstore from documents.
    Saves the index for future use.
    """
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    save_path = f"{documents[0].metadata['source']}_{save_path}"
    vectorstore.save_local(save_path)
    return vectorstore

def create_ensemble_retriever(bm25_retriever, faiss_retriever):
    """
    Combines BM25 and FAISS retrievers into an ensemble retriever.
    """
    return EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever])

def load_faiss_index(index_name, embedding_type="openai", base_save_path="faiss_indexes"):
    """
    Loads a saved FAISS index with correct embedding type.
    """
    embeddings = get_embeddings(embedding_type)
    save_path = os.path.join(base_save_path, index_name)
    
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"No FAISS index found at {save_path}")
    
    return FAISS.load_local(
        save_path, 
        embeddings,
        allow_dangerous_deserialization=True
    )

def initialize_faiss_databases(documents_dict, base_save_path="faiss_indexes"):
    """
    Initialize FAISS databases for each document if they don't exist.
    
    Args:
        documents_dict: Dictionary of {document_name: documents}
        base_save_path: Base directory to save FAISS indexes
    """
    os.makedirs(base_save_path, exist_ok=True)
    embeddings = get_embeddings()
    
    for doc_name, docs in documents_dict.items():
        save_path = os.path.join(base_save_path, f"{doc_name.split('.')[0]}_index")
        # Check if index already exists
        if os.path.exists(save_path):
            print(f"Vector Store already exists for {doc_name}")
        else:
            if doc_name == "brka_trades.xlsx":
                vectorstore = _build_excel_vector_store(chunk_trades())
            elif doc_name == "buffet_qna.xlsx":
                vectorstore = _build_excel_vector_store(chunk_qna_data())
            else:
                vectorstore = FAISS.from_documents(docs, embeddings)

            print(f"Creating new FAISS index for {doc_name}...")
            vectorstore.save_local(save_path)

def initialise_multiple_bm25_retrievers(documents_dict, base_save_path="bm25_indexes"):
    """
    Creates and saves BM25 retrievers for multiple document sets.
    """
    os.makedirs(base_save_path, exist_ok=True)
    retrievers = {}
    
    for doc_name, docs in documents_dict.items():
        # Convert dictionary items to Document objects
        doc_objects = []
        
        # Handle dictionary-type documents
        if isinstance(docs, dict):
            for key, text in docs.items():
                doc_objects.append(
                    Document(
                        page_content=str(text),
                        metadata={
                            "source": doc_name,
                            "key": key
                        }
                    )
                )
        # Handle list-type documents
        elif isinstance(docs, list):
            doc_objects = [
                Document(
                    page_content=str(text),
                    metadata={"source": doc_name}
                ) for text in docs
            ]
        
        save_name = f"{doc_name.split('.')[0]}_bm25.pkl"
        save_path = os.path.join(base_save_path, save_name)
        
        # Create and save retriever
        bm25_retriever = BM25Retriever.from_documents(doc_objects)
        bm25_retriever.k = 5
        
        try:
            with open(save_path, 'wb') as f:
                pickle.dump(bm25_retriever, f)
            print(f"Saved BM25 retriever for {doc_name}")
            retrievers[doc_name] = bm25_retriever
        except Exception as e:
            print(f"Error saving BM25 retriever for {doc_name}: {str(e)}")
    
    return retrievers

def load_bm25_retriever(index_name):
    """
    Loads a BM25 retriever from a saved index file.
    """
    mapping = {
        "trades": "brka_trades_bm25.pkl",
        "qna": "buffet_qna_bm25.pkl",
        "news": "news_bm25.pkl",
        "shareholder_letters": "shareholder_letters_bm25.pkl"
    }
    with open(os.path.join("bm25_indexes", mapping[index_name]), 'rb') as f:
        return pickle.load(f)   


def _build_excel_vector_store(chunks):
    # Convert dictionary chunks to a list of text chunks for embedding
    if isinstance(chunks, dict):
        text_chunks = list(chunks.values())
    else:
        text_chunks = chunks

    # Initialize LangChain's SentenceTransformer Embeddings
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create a FAISS vector store using the custom chunks and the embedding model
    vectorstore = FAISS.from_texts(text_chunks, embedding_model)
    return vectorstore


def main_intialise_retrievers():
    """
    Initialise Vector Stores for all documents 
    Initialise BM25 retrievers for all documents
    """
    documents_dict = {
        "brka_trades.xlsx": chunk_trades(),
        "buffet_qna.xlsx": chunk_qna_data(),
        "news.pkl" : RecursiveCharacterTextSplitter().split_documents(load_news_pickle("data/news.pkl")),
        "shareholder_letters.pkl" : RecursiveCharacterTextSplitter().split_documents(load_shareholder_letters_pickle("data/shareholder_letters.pkl"))
    }
    initialize_faiss_databases(documents_dict)
    bm25_retrievers = initialise_multiple_bm25_retrievers(documents_dict)
    return bm25_retrievers

def main_retrieval_agent(data_sources: List[str], query: str, top_k: int = 5):
    """
    Retrieves documents using correct embedding types for each source.
    """
    retriever_mapping = {
        # Excel-based sources use sentence transformers
        "trades": load_faiss_index("brka_trades_index", embedding_type="sentence_transformers"),
        "qna": load_faiss_index("buffet_qna_index", embedding_type="sentence_transformers"),
        # Other sources use OpenAI
        "news": load_faiss_index("news_index", embedding_type="openai"),
        "shareholder_letters": load_faiss_index("shareholder_letters_index", embedding_type="openai")
    }
    
    # Initialize retrievers for each requested data source
    ensemble_retrievers = []

    for source in data_sources:
        try:
            # Load both FAISS and BM25 retrievers for this source
            faiss_retriever = retriever_mapping[source].as_retriever(
                search_kwargs={"k": top_k * 2}  # Get more docs initially for better reranking
            )
            bm25_retriever = load_bm25_retriever(index_name=source)
            
            # Create ensemble retriever for this source
            ensemble = EnsembleRetriever(
                retrievers=[faiss_retriever, bm25_retriever],
                weights=[0.5, 0.5]  # Equal weights for both retrievers
            )
            ensemble_retrievers.append(ensemble)
            
        except KeyError:
            print(f"Warning: Source '{source}' not found in retriever mappings")
        except Exception as e:
            print(f"Error loading retrievers for {source}: {str(e)}")
    
    if not ensemble_retrievers:
        raise ValueError("No valid retrievers could be loaded")
    
    # Get documents from all ensemble retrievers
    all_docs = []
    for retriever in ensemble_retrievers:
        docs = retriever.get_relevant_documents(query)
        all_docs.extend(docs)
    
    # Remove duplicates (if any)
    unique_docs = _remove_duplicates(all_docs)
    
    # Rerank using cross-encoder
    reranked_docs = rerank_documents(query, unique_docs, top_k=top_k)
    
    # Ensure documents have proper metadata for RAG
    final_docs = _prepare_for_rag(reranked_docs)
    
    return final_docs

def _remove_duplicates(docs: List[Document]) -> List[Document]:
    """
    Remove duplicate documents based on content.
    """
    seen_contents = set()
    unique_docs = []
    
    for doc in docs:
        if doc.page_content not in seen_contents:
            seen_contents.add(doc.page_content)
            unique_docs.append(doc)
    
    return unique_docs

def _prepare_for_rag(docs: List[Document]) -> List[Document]:
    """
    Ensure documents have proper metadata for RAG.
    """
    prepared_docs = []
    for doc in docs:
        # Ensure each document has required metadata
        metadata = doc.metadata.copy() if doc.metadata else {}
        metadata.update({
            "score": getattr(doc, "score", None),  # Preserve reranking score if exists
            "source": metadata.get("source", "unknown"),
            "document_type": metadata.get("document_type", "unknown")
        })
        
        prepared_docs.append(
            Document(
                page_content=doc.page_content,
                metadata=metadata
            )
        )
    
    return prepared_docs
