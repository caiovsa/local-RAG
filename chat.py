import sys
from typing import List, Dict
from openai import OpenAI
from pymilvus import connections, Collection
from config import settings

# Global variables for connections
client = None
collection = None

def initialize_connections():
    """Incia openAI e milvus"""
    global client, collection
    
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    
    # Conexão com milvus
    connections.connect("default", host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)
    
    # Load da coleção
    try:
        collection = Collection(settings.COLLECTION_NAME)
        collection.load()
        print(f"Connected to collection '{settings.COLLECTION_NAME}'")
        return True
    except Exception as e:
        print(f"Error connecting to collection: {e}")
        print("Please run the vectorizer first to create the collection")
        return False

def generate_query_embedding(query: str) -> List[float]:
    """Faz o embedding da pergunta (query) usando OpenAI. Precisamos do embedding para buscar documentos similares."""
    response = client.embeddings.create(input=[query],model="text-embedding-3-large")
    return response.data[0].embedding

def search_similar_documents(query: str, top_k: int = 5) -> List[Dict]:
    """Pega o embedding da pergunta(query) e faz a busca usando IP (Inner Product)."""
    query_embedding = generate_query_embedding(query)
    
    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
    
    results = collection.search(
        data=[query_embedding],
        anns_field="vector",
        param=search_params,
        limit=top_k,
        output_fields=["text", "file_name", "page_number"]
    )
    
    similar_docs = []
    for hit in results[0]:
        similar_docs.append({
            "text": hit.entity.get("text"),
            "file_name": hit.entity.get("file_name"),
            "page_number": hit.entity.get("page_number"),
            "score": hit.score
        })
    
    return similar_docs

def generate_response(query: str, context_docs: List[Dict]) -> str:
    """Função da OPENAI para gerar a resposta baseada nos documentos similares encontrados."""
    # Contexto
    context_text = "\n\n".join([
        f"From {doc['file_name']} (page {doc['page_number']}):\n{doc['text']}"
        for doc in context_docs
    ])
    
    # USER_Prompt
    prompt = f"""You are a helpful AI assistant that answers questions based on PDF documents.
    Use the following context from the documents to answer the user's question. If the answer is not in the context, say so.

    Context:
    {context_text}

    Question: {query}

    Answer:"""

    system = "You are a helpful assistant that answers questions based on provided document context."
    
    
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        max_tokens=30000,
        temperature=0.4
    )
    
    return response.choices[0].message.content

def chat_loop():
    """Main chat loop."""
    print("\nPDF Chat Assistant")
    print("=" * 50)
    print("Ask questions about your PDF documents. Type 'quit' to exit.")
    print()
    
    while True:
        try:
            query = input("You: ").strip()
            
            if query.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            print("Searching for relevant documents...")
            
            # Search for similar documents
            similar_docs = search_similar_documents(query, top_k=3)
            
            if not similar_docs:
                print("No relevant documents found for your query.")
                continue
            
            print(f"Found {len(similar_docs)} relevant document(s)")
            
            # Generate response
            print("Generating response...")
            response = generate_response(query, similar_docs)
            
            print(f"\nAssistant: {response}\n")
            
            # Show sources
            print("Sources:")
            for i, doc in enumerate(similar_docs, 1):
                print(f"  {i}. {doc['file_name']} (page {doc['page_number']}) - Score: {doc['score']:.3f}")
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Main function to run the chat interface."""
    # Initialize connections
    if not initialize_connections():
        sys.exit(1)
    
    # Start chat loop
    chat_loop()

if __name__ == "__main__":
    main()
