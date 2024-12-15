from openai import AzureOpenAI
from typing import List, Optional

class OpenAIHandler:
    def __init__(self, azure_endpoint: str, azure_api_key: str):
        self.azure_client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            api_version="2024-02-15-preview"
        )

    def generate_response(self, query: str, context: List[str]) -> Optional[str]:
        context_str = "\n".join(context)
        prompt = f"Context:\n{context_str}\n\nQuery: {query}\n\nResponse:"
        
        try:
            response = self.azure_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant using provided context to answer queries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating response: {e}")
            return None
