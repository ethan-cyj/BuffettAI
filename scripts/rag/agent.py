import json
import requests
from typing import Optional, List
from googlesearch import search
from bs4 import BeautifulSoup
from langchain.schema import Document
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

from text_splitter import RecursiveCharacterTextSplitter
from retrieval import load_faiss_index, main_retrieval_agent
from llm_config.rag_tools import tools
load_dotenv()

client = OpenAI()

def function_calling_agent(query: str):
    """
    Call the function-calling agent to determine the most suitable data sources for the query.
    """
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": query}],
    tools=tools,
    )

    args = [call.function.arguments for call in response.choices[0].message.tool_calls]
    print(args)
    return args

def retrieval_agent(query: str, data_sources: Optional[List[str]] = None):
    """
    Returns a list of documents from a given vector store.
    If no data sources are provided, it will search the web for the most relevant documents.
    """
    if data_sources is None:
        # call web source agent to retrieve documents
        documents = main_search_agent(query)
        if documents is None:
            return None
        else:
            return documents
    else:
        # Retrieve from vector store
        search_sources =[json.loads(data_source).get("data_source") for data_source in data_sources]
        documents = main_retrieval_agent(query=query, data_sources=search_sources)
        if documents is None:
            # call web source agent to retrieve documents
            documents = main_search_agent(query)
            if documents is None:
                return None
            else:
                return documents    
        return documents
def _google_search(query, num_results=5, lang='en'):
    """
    Perform a Google search and return a list of URLs.
    
    Parameters:
        query (str): The search query.
        num_results (int): The maximum number of results to return.
        lang (str): The language for search results.
    
    Returns:
        List[str]: A list of result URLs.
    """
    return list(search(query, num_results=num_results, lang=lang))

def _fetch_html(url):
    """
    Retrieve the HTML content of the given URL.
    
    Parameters:
        url (str): The URL to fetch.
        
    Returns:
        str: The HTML content if the request is successful, else None.
    """
    try:
        # Set a user-agent to mimic a browser
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
        return response.text
    except requests.RequestException as e:
        print(f"Failed to retrieve {url}: {e}")
        return None
    
def clean_html_content(html):
    """
    Clean and extract relevant text from HTML content.
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove unwanted elements
    for element in soup(['script', 'style', 'nav', 'header', 'footer', 'ads']):
        element.decompose()
    
    # Extract main content (adjust selectors based on common website structures)
    main_content = soup.find('main') or soup.find('article') or soup.find('body')
    
    if main_content:
        # Get text and clean it
        text = main_content.get_text(separator=' ', strip=True)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    return None

def process_html_to_documents(url, html):
    """
    Convert HTML content to structured documents for RAG.
    """
    # Clean the HTML
    clean_text = clean_html_content(html)
    if not clean_text:
        return []

    # Create documents with metadata
    chunks = RecursiveCharacterTextSplitter().create_documents(
        texts=[clean_text],
        metadatas=[{
            "source": url,
            "type": "web_content",
            "timestamp": datetime.now().isoformat()
        }]
    )
    
    return chunks

def main_search_agent(query):
    """
    Search and process web content for RAG when local data is insufficient.
    Returns a list of processed documents ready for RAG.
    """
    all_documents = []
    urls = _google_search(query)
    
    for url in urls:
        html = _fetch_html(url)
        if html:
            try:
                documents = process_html_to_documents(url, html)
                if documents:
                    all_documents.extend(documents)
                    print(f"Processed {len(documents)} chunks from {url}")
                else:
                    print(f"No useful content extracted from {url}")
                    return None
            except Exception as e:
                print(f"Error processing {url}: {str(e)}")
                return None
        else:
            print(f"Could not retrieve HTML for {url}")
            return None
    
    return all_documents


def main_routing_function(query: str):
    """
    Main routing function that orchestrates the retrieval of documents from vector store or web search.
    """
    data_sources = function_calling_agent(query)
    return retrieval_agent(query, data_sources)

