from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

FUNCTION_DESCRIPTION = """
Identify the suitable data sources for the given question. There are 4 data sources:
- trades: Berkshire Hathaway's trades
- qna: Warren Buffett's Q&A
- news: News articles regarding Berkshire Hathaway and Warren Buffett
- letters: Shareholder letters from Warren Buffett

The function should return a list of data sources that are most suitable for the given question.
"""


tools = [
    {
        "type": "function",
        "function": {
            "name": "most_suitable_data_source",
            "description": FUNCTION_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "data_source": {
                        "type": "string",   
                        "description": "Identifier of the data source from which to retrieve data: \n - trades \n - qna \n - news \n - letters"
                    },
                },
                "required": [
                    "data_source",
                ],
                "additionalProperties": False
            },
            "strict": True
        }
    }
]
