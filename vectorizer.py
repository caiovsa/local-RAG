import os
import time
from pathlib import Path
from typing import List, Dict
from openai import OpenAI
from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, 
    DataType, MilvusClient, utility
)
from config import settings
from utils import extract_text_from_pdf, split_text, generate_doc_id, clean_text

# Global variables
client = None
milvus_client = None
collection = None

def initialize_connections():
    """OPENAI e Milvus connections"""
    global client, milvus_client
    
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    milvus_client = MilvusClient(settings.MILVUS_URI)
    
    # Milvus
    connections.connect("default", host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)
    print("✅ Connected to Milvus")

def setup_collection():
    """Cria ou recria a coleção no Milvus. (Dropa se já existir para poder inserir novos dados)"""
    global collection
    
    if utility.has_collection(settings.COLLECTION_NAME):
        print(f"Collection '{settings.COLLECTION_NAME}' already exists - dropping it...")
        utility.drop_collection(settings.COLLECTION_NAME)
        print(f"✅ Collection '{settings.COLLECTION_NAME}' dropped successfully")
    
    print(f"Creating new collection '{settings.COLLECTION_NAME}'")
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=settings.EMBEDDING_DIM),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="page_number", dtype=DataType.INT64),
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=512),
    ]
    
    schema = CollectionSchema(fields=fields, description="PDF document embeddings")
    collection = Collection(settings.COLLECTION_NAME, schema)
    
    # Index
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "IP",
        "params": {"nlist": 1024}
    }
    collection.create_index(field_name="vector", index_params=index_params)
    print(f"✅ Collection '{settings.COLLECTION_NAME}' created successfully")
    
    # Load collection
    collection.load()

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Embeddings são gerados usando o modelo text-embedding-3-large do OpenAI.
    Eles são usados para representar o texto em um espaço vetorial."""
    embeddings = []
    batch_size = 100  # Processa em batches para evitar limites do OpenAI API
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"Generating embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        response = client.embeddings.create(
            input=batch,
            model="text-embedding-3-large"
        )
        
        batch_embeddings = [data.embedding for data in response.data]
        embeddings.extend(batch_embeddings)
        
        # Delay para evitar o hitting API rate limits
        time.sleep(0.5)
    
    return embeddings

def process_pdf(pdf_path: str) -> int:
    """Processa um unico PDF, extrai texto, divide em chunks, gera embeddings e insere no Milvus."""
    print(f"Processing PDF: {pdf_path}")
    
    # Extrai texto do PDF
    pages = extract_text_from_pdf(pdf_path)
    if not pages:
        print(f"No text found in {pdf_path}")
        return 0
    
    # Processa paginas em chunks
    all_chunks = []
    all_metadata = []
    
    for page in pages:
        text = clean_text(page["content"])
        chunks = split_text(text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        
        for chunk_idx, chunk in enumerate(chunks):
            chunk_id = f"{generate_doc_id(page['content'])}_{chunk_idx}"
            all_chunks.append(chunk)
            all_metadata.append({
                "file_name": page["file_name"],
                "page_number": page["page_number"],
                "chunk_id": chunk_id
            })
    
    if not all_chunks:
        print(f"No valid chunks found in {pdf_path}")
        return 0
    
    # embeddings
    print(f"Generating embeddings for {len(all_chunks)} chunks...")
    embeddings = generate_embeddings(all_chunks)
    
    # Prepara os dados para a inserção no Milvus
    data_to_insert = []
    for i, (chunk, embedding, metadata) in enumerate(zip(all_chunks, embeddings, all_metadata)):
        data_to_insert.append({
            "vector": embedding,
            "text": chunk,
            "file_name": metadata["file_name"],
            "page_number": metadata["page_number"],
            "chunk_id": metadata["chunk_id"]
        })
    
    # Insere no Milvus
    print(f"Inserting {len(data_to_insert)} chunks into Milvus...")
    collection.insert(data_to_insert)
    
    print(f"Successfully processed {pdf_path} - {len(data_to_insert)} chunks inserted")
    return len(data_to_insert)

def process_all_pdfs() -> int:
    """Processa todos os PDFs no diretório configurado."""
    pdf_dir = Path(settings.PDF_DIRECTORY)
    if not pdf_dir.exists():
        print(f"Creating PDF directory: {pdf_dir}")
        pdf_dir.mkdir(exist_ok=True)
        print(f"Please add PDF files to {pdf_dir} and run again")
        return 0
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return 0
    
    total_chunks = 0
    for pdf_file in pdf_files:
        chunks = process_pdf(str(pdf_file))
        total_chunks += chunks
    
    print(f"\nProcessing complete! Total chunks inserted: {total_chunks}")
    return total_chunks

def main():
    """Main function."""
    print("Starting Local PDF Vectorizer")
    print("=" * 50)
    
    # Initialize connections
    initialize_connections()
    
    # Setup collection (drops existing if present)
    setup_collection()
    
    # Process all PDFs
    process_all_pdfs()

if __name__ == "__main__":
    main()
