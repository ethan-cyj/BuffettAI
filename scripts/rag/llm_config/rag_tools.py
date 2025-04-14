from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

FUNCTION_DESCRIPTION = """
Identify the most suitable data source for the given question and decide which language model should be used.
There are 4 data sources:
- trades: Berkshire Hathaway's trades,
- qna: Warren Buffett's Q&A,
- news: News articles regarding Berkshire Hathaway and Warren Buffett,
- shareholder_letters: Shareholder letters from Warren Buffett.
Return none if no data source is found.

Additionally, if the question is personal to Warren Buffett, and not work related, use 'ollama';
otherwise, use 'openai'.

Return a JSON object with two keys:
- "data_source": a string that specifies the most relevant data source, default to 'none'
- "model": either "ollama" or "openai" based on the query's complexity. default to 'openai
"""


tools = [
    {
        "type": "function",
        "function": {
            "name": "most_suitable_data_source_and_model",
            "description": FUNCTION_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "data_source": {
                        "type": "string",   
                        "description": "Identifier of the data source from which to retrieve data: \n - trades \n - qna \n - news \n - shareholder_letters. \n - none (if no data source is found)"
                    },
                    "model": {
                        "type": "string",
                        "enum": ["openai", "ollama"],
                        "description": "The model to use for the query. Use 'ollama' for simple, non-technical, non-work related, personal, nonsense questions and 'openai' for the rest."
                    }
                },
                "required": [
                    "data_source",
                    "model"
                ],
                "additionalProperties": False
            },
            "strict": True
        }
    }
]
