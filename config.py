import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # VM
    VM_MODEL = "ibm-granite/granite-3.3-8b-instruct"
    VM_ADDRESS = "https://4gca2nseftuoyo-8000.proxy.runpod.net/v1"
    
    # Milvus (local)
    MILVUS_HOST = "localhost"
    MILVUS_PORT = "19530"
    MILVUS_URI = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
    
    # Collection settings
    COLLECTION_NAME = "pdf_documents"
    EMBEDDING_DIM = 3072  # text-embedding-3-large dimension
    
    # PDF processing
    PDF_DIRECTORY = "pdfs"
    CHUNK_SIZE = 600
    CHUNK_OVERLAP = 150

settings = Settings()
