import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # VM
    VM_MODEL = "ibm-granite/granite-3.3-8b-instruct"
    VM_ADDRESS = os.getenv("VM_ADDRESS")
    LOCAL_VM_ADDRESS = os.getenv("LOCAL_VM_ADDRESS")
    
    # Milvus (local)
    MILVUS_HOST = "localhost"
    MILVUS_PORT = "19530"
    MILVUS_URI = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
    
    # Collection 
    COLLECTION_NAME = "pdf_documents"
    EMBEDDING_DIM = 3072  # text-embedding-3-large dimension
    
    # PDF 
    PDF_DIRECTORY = "pdfs"
    CHUNK_SIZE = 600
    CHUNK_OVERLAP = 150

settings = Settings()
