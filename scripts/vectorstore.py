import os
import time
from typing import List
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from data_utils import load_all_documents, load_documents

load_dotenv()

def batch_process_documents(docs: List, batch_size: int = 50, delay: float = 1.0):
    """
    Process documents in batches with delay between batches to avoid rate limits.
    
    Args:
        docs: List of documents to process
        batch_size: Number of documents per batch
        delay: Delay in seconds between batches
    """
    embeddings = OpenAIEmbeddings()
    all_embeddings = []
    
    # Calculate number of batches for progress bar
    num_batches = (len(docs) + batch_size - 1) // batch_size
    
    print(f"Processing {len(docs)} documents in {num_batches} batches...")
    
    for i in tqdm(range(0, len(docs), batch_size)):
        batch = docs[i:i + batch_size]
        try:
            # Create temporary FAISS index for this batch
            temp_vectorstore = FAISS.from_documents(batch, embeddings)
            all_embeddings.append(temp_vectorstore)
            
            # Add delay between batches
            if i + batch_size < len(docs):  # Don't delay after the last batch
                time.sleep(delay)
                
        except Exception as e:
            print(f"Error in batch {i//batch_size}: {str(e)}")
            print("Increasing delay and retrying...")
            time.sleep(delay * 2)  # Double delay on error
            try:
                temp_vectorstore = FAISS.from_documents(batch, embeddings)
                all_embeddings.append(temp_vectorstore)
            except Exception as e:
                print(f"Retry failed: {str(e)}")
                continue
    
    # Merge all embeddings into one FAISS index
    if not all_embeddings:
        raise Exception("No embeddings were created successfully")
    
    final_index = all_embeddings[0]
    for index in all_embeddings[1:]:
        final_index.merge_from(index)
    
    return final_index

def initialize_all_faiss_stores(data_dir="../data", save_dir="faiss_indexes", batch_size=50):
    """
    Create FAISS vectorstores for all documents in the data directory with batching.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("Loading documents from", data_dir)
    raw_docs = load_all_documents(base_path=data_dir)
    
    for filename, docs in raw_docs.items():
        if filename != "news.pkl":  # Keeping your existing filter
            continue
            
        index_name = os.path.splitext(filename)[0]
        save_path = os.path.join(save_dir, f"{index_name}_index")
        
        if os.path.exists(save_path):
            print(f"FAISS index already exists for {filename}, skipping...")
            continue
            
        print(f"Creating FAISS index for {filename}...")
        try:
            # Process documents in batches
            vectorstore = batch_process_documents(docs, batch_size=batch_size)
            vectorstore.save_local(save_path)
            print(f"Successfully created and saved index for {filename}")
        except Exception as e:
            print(f"Error creating index for {filename}: {str(e)}")

if __name__ == "__main__":
    DATA_DIR = "./data"
    SAVE_DIR = "faiss_indexes"
    BATCH_SIZE = 500  # Adjust this based on your rate limits
    
    print("Starting FAISS index initialization...")
    initialize_all_faiss_stores(DATA_DIR, SAVE_DIR, BATCH_SIZE)
    print("Finished creating all FAISS indexes!")