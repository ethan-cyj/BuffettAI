import numpy as np
from sentence_transformers import CrossEncoder

# Initialize the pre-trained cross-encoder (adjust model name as needed)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_documents(query, documents, top_k=5):
    """
    Reranks a list of document objects based on relevance to the query.
    Each document is assumed to have a 'content' field.
    """
    pairs = [[query, doc["content"]] for doc in documents]
    scores = cross_encoder.predict(pairs)
    ranked_indices = np.argsort(scores)[::-1][:top_k]
    return [documents[i] for i in ranked_indices]
