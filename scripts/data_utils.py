import os
import pandas as pd
import pickle
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
import warnings
from pathlib import Path
from pprint import pprint

def safe_load_pickle(file_path: str) -> Any:
    """Safely load pickle files that may contain pandas objects"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            # First try standard pickle load
            with open(file_path, "rb") as f:
                return pickle.load(f)
        except (TypeError, AttributeError, pickle.UnpicklingError):
            try:
                # Fallback to pandas read_pickle
                return pd.read_pickle(file_path)
            except Exception as e:
                raise ValueError(f"Failed to load {file_path}: {str(e)}")

def inspect_file(file_path: str) -> None:
    """Debug function to examine file structure"""
    print(f"\n=== Inspecting {file_path} ===")
    try:
        if file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
            print("Excel file detected")
            print(f"Shape: {df.shape}")
            print("Columns:", df.columns.tolist())
            print("\nFirst row:")
            pprint(df.iloc[0].to_dict())
        elif file_path.endswith('.pkl'):
            data = safe_load_pickle(file_path)
            print(f"Type: {type(data)}")
            if isinstance(data, pd.DataFrame):
                print("Pandas DataFrame detected")
                print(f"Shape: {data.shape}")
                print("Columns:", data.columns.tolist())
                print("\nFirst row:")
                pprint(data.iloc[0].to_dict())
            elif isinstance(data, list):
                print(f"List of {len(data)} items")
                if data:
                    print("\nFirst item type:", type(data[0]))
                    if isinstance(data[0], dict):
                        print("Keys in first item:", data[0].keys())
            elif isinstance(data, dict):
                print("Dictionary with keys:", data.keys())
            else:
                print("Content sample:", str(data)[:200] + "...")
        else:
            raise ValueError("Unsupported file format")
    except Exception as e:
        print(f"Inspection failed: {str(e)}")

def load_buffet_qna_xlsx(file_path: str) -> List[Document]:
    """
    Load Q&A data from Excel with columns: Section, Questions, Answers
    Returns List[Document] where:
    - page_content contains formatted Q&A
    - metadata contains structured fields
    """
    df = pd.read_excel(file_path)
    
    # Validate required columns
    required_cols = {'Section', 'Questions', 'Answers'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in Q&A file: {missing}")

    documents = []
    for _, row in df.iterrows():
        # Create human-readable content
        page_content = (
            f"Question: {row['Questions']}\n"
            f"Answer: {row['Answers']}\n"
            f"Section: {row['Section']}"
        )
        
        # Store structured data in metadata
        metadata = {
            "section": row["Section"],
            "question": row["Questions"],
            "answer": row["Answers"],
            "source": "buffet_qna"
        }
        
        documents.append(Document(
            page_content=page_content,
            metadata=metadata
        ))
    
    return documents

def load_brka_trades_xlsx(file_path: str) -> List[Document]:
    """
    Load trades data from Excel with financial columns
    Returns List[Document] where:
    - page_content contains key trade info
    - metadata contains all raw data
    """
    df = pd.read_excel(file_path)
    
    # Validate required columns
    required_cols = {'RIC', 'Security Name', 'Date', 'Position', 'Position Change'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in trades file: {missing}")

    documents = []
    for _, row in df.iterrows():
        # Create human-readable summary
        page_content = (
            f"Security: {row['Security Name']} ({row['RIC']})\n"
            f"Date: {row['Date']}\n"
            f"Position: {row['Position']:,} shares\n"
            f"Change: {row['Position Change']:+,}"
        )
        
        # Store all raw data in metadata
        metadata = row.to_dict()
        metadata.update({"source": "brka_trades"})
        
        documents.append(Document(
            page_content=page_content,
            metadata=metadata
        ))
    
    return documents

def load_news_pickle(file_path: str) -> List[Document]:
    """Load news pickle with flexible format handling"""
    data = safe_load_pickle(file_path)
    
    # Handle DataFrame case
    if isinstance(data, pd.DataFrame):
        data = data.to_dict('records')
    
    # Handle list of dicts
    if isinstance(data, list) and all(isinstance(x, dict) for x in data):
        return [
            Document(
                page_content=item.get('text', item.get('content', str(item))),
                metadata={k: v for k, v in item.items() 
                         if k not in ['text', 'content']}
            )
            for item in data
        ]
    
    # Handle single dictionary
    elif isinstance(data, dict):
        return [Document(
            page_content=data.get('text', data.get('content', str(data))),
            metadata={k: v for k, v in data.items() 
                     if k not in ['text', 'content']}
        )]
    
    # Fallback for other formats
    return [Document(page_content=str(data))]

def load_shareholder_letters_pickle(file_path: str) -> List[Document]:
    """Load shareholder letters with year-based structure"""
    data = safe_load_pickle(file_path)
    
    if isinstance(data, dict) and all(isinstance(k, (str, int)) for k in data):
        return [
            Document(
                page_content=content,
                metadata={"year": year, "source": "shareholder_letter"}
            )
            for year, content in data.items()
        ]
    
    # Fallback for other formats
    return load_news_pickle(file_path)

def load_documents(file_path: str) -> List[Document]:
    """Main document loading interface"""
    try:
        if file_path.endswith('.xlsx'):
            if 'buffet_qna' in file_path.lower():
                return load_buffet_qna_xlsx(file_path)
            elif 'brka_trades' in file_path.lower():
                return load_brka_trades_xlsx(file_path)
        
        elif file_path.endswith('.pkl'):
            if 'news' in file_path.lower():
                return load_news_pickle(file_path)
            elif 'shareholder' in file_path.lower():
                return load_shareholder_letters_pickle(file_path)
        
        raise ValueError(f"Unrecognized file type: {file_path}")
    
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return [Document(
            page_content=f"Error loading document: {str(e)}",
            metadata={"source": "error", "file_path": file_path}
        )]

def load_all_documents(
    base_path: str = "/data",
    buffet_qna_path: Optional[str] = None,
    brka_trades_path: Optional[str] = None,
    news_path: Optional[str] = None,
    shareholder_letters_path: Optional[str] = None,
    verbose: bool = True
) -> List[Document]:
    """
    Load all 4 document sources at once.
    
    Args:
        base_path: Base directory for files if individual paths not specified
        *_path: Override individual file paths
        verbose: Print loading progress
    
    Returns:
        Combined list of Documents from all sources
    """
    # Set default paths if not specified
    buffet_qna_path = buffet_qna_path or Path(base_path) / "buffet_qna.xlsx"
    brka_trades_path = brka_trades_path or Path(base_path) / "brka_trades.xlsx"
    news_path = news_path or Path(base_path) / "news.pkl"
    shareholder_letters_path = shareholder_letters_path or Path(base_path) / "shareholder_letters.pkl"
    
    all_docs = []
    
    # Load each file with progress reporting
    for file_path, loader in [
        (buffet_qna_path, load_buffet_qna_xlsx),
        (brka_trades_path, load_brka_trades_xlsx),
        (news_path, load_news_pickle),
        (shareholder_letters_path, load_shareholder_letters_pickle)
    ]:
        try:
            if verbose:
                print(f"Loading {file_path}...")
            docs = loader(file_path)
            all_docs.extend(docs)
            if verbose:
                print(f"Loaded {len(docs)} documents")
        except Exception as e:
            if verbose:
                print(f"Error loading {file_path}: {str(e)}")
            all_docs.append(Document(
                page_content=f"Error loading {file_path}: {str(e)}",
                metadata={"source": "error", "file_path": str(file_path)}
            ))
    
    if verbose:
        print(f"\nTotal documents loaded: {len(all_docs)}")
    
    return all_docs


# print(f"Script running from: {os.getcwd()}")

# print(load_documents("../data/news.pkl"))
# print(load_all_documents("../data"))