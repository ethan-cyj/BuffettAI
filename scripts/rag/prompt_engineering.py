def create_prompt(company_name: str, documents):
    """
    Creates a prompt for the LLM using the retrieved documents.
    Assumes each document is a dict with a 'content' field.
    """
    context = "\n".join([doc["content"] for doc in documents])
    prompt = f"""
You are a financial analyst tasked with generating an investment report. Use the following context:
{context}

Focus on revenue growth, profitability, and initiatives.
Provide specific numbers and facts where available.
Use a professional, concise tone.
Structure the report with these sections:
- Overview
- Revenue Growth
- Profitability
- ESG Initiatives
- Conclusion

Investment Report for {company_name}:
"""
    return prompt

def create_evaluation_prompt(query: str, generated_report: str, retrieved_documents):
    """
    Creates an evaluation prompt for the LLM to judge the generated report.
    """
    retrieved_context = "\n".join([doc["content"] for doc in retrieved_documents])
    prompt = f"""
You are an expert financial analyst evaluating an investment report.
    
Query:
{query}

Generated Report:
{generated_report}

Retrieved Context:
{retrieved_context}

Evaluation Criteria:
1. Relevance (0-10): Does the report address the query?
2. Accuracy (0-10): Are the facts correct and supported by the retrieved context?
3. Coherence (0-10): Is the report well-structured and easy to understand?

Provide a score for each criterion along with a brief explanation.
"""
    return prompt
