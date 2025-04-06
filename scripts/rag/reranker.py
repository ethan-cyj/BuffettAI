import numpy as np
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document



# Initialize the pre-trained cross-encoder (adjust model name as needed)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_documents(query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
    """
    Reranks documents using a cross-encoder model.
    
    Args:
        query: The search query
        documents: List of LangChain Document objects to rerank
        top_k: Number of top documents to return
        
    Returns:
        List of reranked Document objects
    """
    # Create pairs of (query, document_content)
    pairs = [[query, doc.page_content] for doc in documents]  # Changed to use page_content
    
    # Get scores from cross-encoder
    scores = cross_encoder.predict(pairs)
    
    # Sort by scores in descending order and get top_k indices
    ranked_indices = np.argsort(scores)[::-1][:top_k]
    
    # Return documents in ranked order
    return [documents[i] for i in ranked_indices]