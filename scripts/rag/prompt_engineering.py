from typing import List, Dict, Any, Optional, Literal
from langchain_core.documents import Document   
from pydantic import BaseModel, Field

def create_prompt(query: str,  documents: List[Document]) -> str:
    """
    Creates a focused prompt for the LLM using both the query and retrieved documents.
    
    Args:
        query: The original user query/search terms
        company_name: Target company for the report
        documents: List of retrieved LangChain Document objects
        
    Returns:
        Formatted prompt string
    """
    context = ""
    if documents:
        context = "\n".join([f"Source {i+1}:\n{doc.page_content}\n" 
                        for i, doc in enumerate(documents)])
    
    prompt = f"""
**Task**: You are Warren Buffett, the CEO of Berkshire Hathaway. Use only the docuements provided to answer the question. Answer the question in a first person perspective.: 
"{query}"

Context:
{context}
"""
    return prompt.strip()

def create_evaluation_prompt(query: str, generated_report: str, 
                           retrieved_documents: List[Document]) -> str:
    """
    Creates an evaluation prompt that assesses how well the report addresses the query.
    
    Args:
        query: Original user question/search terms
        generated_report: LLM-generated report to evaluate
        retrieved_documents: Documents used for context
        
    Returns:
        Formatted evaluation prompt
    """
    context_samples = "\n".join([f"• {doc.page_content[:200]}..." 
                               for doc in retrieved_documents[:3]])
    
    prompt = f"""
**Evaluation Task**: Assess how well this investment report addresses the original query.

**Original Query**: "{query}"

**Generated Report**:
{generated_report}

**Sample Supporting Context**:
{context_samples}

**Evaluation Rubric**:
1. Query Relevance (0-10): 
   - Does every section directly address aspects of "{query}"?
   - Are there any irrelevant tangents?

2. Data Accuracy (0-10):
   - Are all facts supported by the context documents?
   - Are numbers and claims properly qualified?

3. Analytical Depth (0-10):
   - Does it surface non-obvious insights from the data?
   - Does it identify relationships between different metrics?

4. Actionability (0-10):
   - Would an investor find clear takeaways?
   - Are risks and opportunities properly highlighted?

**Output Format**:
- For each criterion: 
  [Score] [Justification in 1-2 sentences]
- Overall summary (1 paragraph)

**Begin Evaluation**:
"""
    return prompt.strip()


    
RAG_EVALUATION_SYSTEM_PROMPT = """
# Task
Your task is to evaluate the quality of a RAG system.

# Instructions
You will be given a report and a set of documents.
You need to evaluate the report based on the given criteria by assigning a score between 1 and 5. 1 is the lowest and 5 is the highest.

# Criteria
- Query Relevance
- Data Accuracy
- Analytical Depth
- Overall Score: This score should be the average of all the scores above.

# Output Format
Return your output in the given JSON format.

Think step by step. I will give you 100 Cookies for doing this task.
"""

class RAGEvaluation(BaseModel):
    reasoning: str = Field(...,
                          description = "Reasoning about the report and the criteria")
    query_relevance: Literal[0, 1, 2, 3] = Field(...,
                                 description = """
                                 How well does the response address the aspects of the query?
                                 Output 3 if the response fully addresses the query, 2 if it addresses the query somewhat well, and 1 if it does not address the query at all. 
                                 Output 0 if the query is not about the report.
                                 """)
    data_accuracy: Literal[0, 1, 2, 3] = Field(...,
                                 description = """
                                 How accurate are the numbers and claims within the response as compared to the context documents?
                                 Output 3 if the response is very accurate, 2 if it is somewhat accurate, and 1 if it is not accurate at all.
                                 Output 0 if the response contains data not supported by the context documents.
                                 Output 3 if the response does not require any accuracy as the query is not about the data itself.
                                 """)
    clarity: Literal[0, 1, 2, 3] = Field(...,
                                 description = """
                                 How clear is the response?
                                 Output 3 if the response can be fully understood without requiring any additional context.
                                 Output 2 if it is somewhat clear - slightly challenging to understand and ambiguous.
                                 Output 1 if it is not clear - requires a lot of additional context to understand.
                                 Output 0 if the response is not clear and cannot be understood by a layman.
                                 """)

