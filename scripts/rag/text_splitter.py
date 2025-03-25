from langchain.text_splitter import RecursiveCharacterTextSplitter


CHUNK_SIZE = 1024
CHUNK_OVERLAP = 20

def split_documents(documents):
    """
    Splits a list of document objects into manageable text chunks.
    Assumes each document has a 'content' field.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    # For compatibility with langchain's splitter, wrap your dicts in a simple object:
    # Here we assume each document is a dict with a 'content' key.
    # You might need to convert these dicts to the Document type expected by langchain.
    return splitter.split_documents(documents)
