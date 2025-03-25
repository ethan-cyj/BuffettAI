import csv
import pickle

def load_documents_from_csv(csv_path, content_column="content", doc_id_column="doc_id"):
    """
    Loads documents from a CSV file and returns a list of dicts with
    keys: 'doc_id' and 'content'.
    """
    documents = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            documents.append({
                "doc_id": row[doc_id_column],
                "content": row[content_column]
            })
    return documents

def load_documents_from_pickle(pickle_path):
    """
    Loads documents from a pickle file.
    """
    with open(pickle_path, "rb") as f:
        documents = pickle.load(f)
    return documents

def save_documents_to_pickle(documents, pickle_path):
    """
    Saves a list of documents to a pickle file.
    """
    with open(pickle_path, "wb") as f:
        pickle.dump(documents, f)
