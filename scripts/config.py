import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

class Config:
    # Milvus connection settings
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
    
    # Azure OpenAI settings
    AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "")
    AZURE_API_KEY = os.getenv("AZURE_API_KEY", "")
    
    # Sentence Transformer model settings
    SENTENCE_MODEL = os.getenv("SENTENCE_MODEL", "all-MiniLM-L6-v2")
