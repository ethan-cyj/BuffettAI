import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter


CHUNK_SIZE = 1024
CHUNK_OVERLAP = 20

def recursive_split_documents(documents):
    """
    Splits a list of document objects into manageable text chunks.
    Assumes each document has a 'content' field.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    # For compatibility with langchain's splitter, wrap your dicts in a simple object:
    # Here we assume each document is a dict with a 'content' key.
    # You might need to convert these dicts to the Document type expected by langchain.
    return splitter.split_documents(documents)

def chunk_qna_data(file_path="data/buffet_qna.xlsx"):
    """
    Reads an Excel file and groups the data by Section,
    where each chunk is a text block containing the question-answer pairs.
    
    Parameters:
        file_path (str): The path to the Excel file.
    
    Returns:
        A dictionary where keys are the unique Section values and 
        values are text strings containing the Q&A pairs for that section.
    """
    # Read the Excel file into a pandas DataFrame
    df = pd.read_excel(file_path)
    
    # Create a dictionary to hold text chunks by section
    chunks = {}
    
    # Group by the 'Section' column
    grouped = df.groupby('Section')
    
    for section, group in grouped:
        text_chunk = ""
        # Create a formatted text for each Q&A pair in the section
        for record in group[['Questions', 'Answers']].to_dict(orient='records'):
            text_chunk += f"Q: {record['Questions']}\nA: {record['Answers']}\n\n"
        chunks[section] = text_chunk.strip()  # Remove any trailing newlines
    
    return chunks

def chunk_trades(file_path="data/brka_trades.xlsx", group_by_column="RIC"):
    """
    Reads an Excel file containing trade data and chunks the data into text blocks.
    
    Parameters:
        file_path (str): The path to the Excel file.
        group_by_column (str, optional): The column to group by (e.g., "RIC"). 
            If None, the function returns a list of row-by-row text chunks.
    
    Returns:
        If group_by_column is provided, returns a dictionary where keys are the unique values 
        in that column and values are text strings that contain the trade records.
        Otherwise, returns a list of text strings (each string represents one trade record).
    """
    # Read the Excel file into a pandas DataFrame
    df = pd.read_excel(file_path)
    
    # If no grouping is specified, return each row as a separate text chunk.
    if group_by_column is None:
        records = df.to_dict(orient='records')
        chunks = []
        for record in records:
            text = (
                f"RIC: {record['RIC']}, Security: {record['Security Name']}, "
                f"Date: {record['Date']}, Weightage: {record['Weightage']}, "
                f"Value: {record['Value']}, Value Change: {record['Value Change']}, "
                f"Position: {record['Position']}, Position Change: {record['Position Change']}\n"
            )
            chunks.append(text)
        return chunks

    # Otherwise, group by the specified column.
    grouped = df.groupby(group_by_column)
    chunks = {}
    for group, data in grouped:
        text_chunk = ""
        for record in data.to_dict(orient='records'):
            text_chunk += (
                f"RIC: {record['RIC']}, Security: {record['Security Name']}, "
                f"Date: {record['Date']}, Weightage: {record['Weightage']}, "
                f"Value: {record['Value']}, Value Change: {record['Value Change']}, "
                f"Position: {record['Position']}, Position Change: {record['Position Change']}\n"
            )
        chunks[group] = text_chunk
    return chunks

