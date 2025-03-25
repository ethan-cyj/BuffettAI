from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

def create_prompt(query: str, company_name: str, documents: List[Document]) -> str:
    """
    Creates a focused prompt for the LLM using both the query and retrieved documents.
    
    Args:
        query: The original user query/search terms
        company_name: Target company for the report
        documents: List of retrieved LangChain Document objects
        
    Returns:
        Formatted prompt string
    """
    context = "\n".join([f"Source {i+1}:\n{doc.page_content}\n" 
                        for i, doc in enumerate(documents)])
    
    prompt = f"""
**Task**: Generate a comprehensive investment report for {company_name} that specifically addresses: 
"{query}"

**Context from company documents**:
{context}

**Report Requirements**:
1. Directly respond to the query about {query}
2. Focus on these key aspects (if relevant):
   - Revenue growth trends and drivers
   - Profitability metrics and margins
   - Strategic initiatives and investments
   - Competitive positioning
   - Risks and challenges
3. Include specific numbers, percentages, and timeframes when available
4. Highlight any contradictions or uncertainties in the data
5. Maintain objective, professional tone

**Report Structure**:
[Overview] - Brief introduction addressing the query
[Key Findings] - Bullet points of most relevant insights
[Detailed Analysis] - Expanded discussion with supporting data
[Conclusions] - Summary and forward-looking statements

**Begin Report for {company_name}**:
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