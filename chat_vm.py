import sys
from typing import List, Dict
from openai import OpenAI
from pymilvus import connections, Collection
from config import settings

# Global variables for connections
openai_client = None
vm_client = None
collection = None

def initialize_connections():
    """Initialize OpenAI client for embeddings and VM client for chat completions"""
    global openai_client, vm_client, collection
    
    # OpenAI client for embeddings
    openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    
    # VM client for chat completions (your local model)
    vm_client = OpenAI(api_key=settings.OPENAI_API_KEY,
        base_url= settings.VM_ADDRESS,  # Local VM model address
    )
    
    # Connection with Milvus
    connections.connect("default", host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)
    
    # Load collection
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
    """Generate embedding for the query using OpenAI"""
    response = openai_client.embeddings.create(
        input=[query],
        model="text-embedding-3-large"
    )
    return response.data[0].embedding

def search_similar_documents(query: str, top_k: int = 5) -> List[Dict]:
    """Get query embedding and search using IP (Inner Product)"""
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
    """Use VM model to generate response with improved error handling"""
    # Context
    context_text = "\n\n".join([
        f"From {doc['file_name']} (page {doc['page_number']}):\n{doc['text']}"
        for doc in context_docs
    ])
    
    # Simplified prompt for better compatibility
    prompt = f"""Baseado no seguinte contexto dos documentos, responda à pergunta do usuário em português.
        Contexto:
        {context_text}
        Pergunta: {query}
        Resposta:"""
    
    system = "Você é um assistente útil chamado CAIO que responde perguntas baseado no contexto de documentos fornecidos. Responda sempre em português."
    
    try:
        # Using VM model for chat completion
        response = vm_client.chat.completions.create(
            model=settings.VM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            extra_body={"chat_template_kwargs": {"thinking": False}}, # Esse é para o granite
            max_tokens=25000,
            temperature=0.3,
            stream=False
        )
        
        # Check if response and choices exist
        if response and hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            message = choice.message
            
            # Handle both content and reasoning_content fields
            if hasattr(message, 'content') and message.content:
                return message.content
            elif hasattr(message, 'reasoning_content') and message.reasoning_content:
                # SGLang with reasoning parser puts response in reasoning_content
                return message.reasoning_content
            else:
                return "Desculpe, não consegui gerar uma resposta. O modelo retornou conteúdo vazio."
        else:
            return "Erro: Resposta inválida do modelo."
            
    except Exception as e:
        print(f"ERROR in generate_response: {e}")
        return f"Erro ao gerar resposta: {str(e)}"


def chat_loop():
    """Main chat loop with improved error handling"""
    print("\nPDF Chat Assistant (VM Model)")
    print("=" * 50)
    print("Ask questions about your PDF documents. Type 'quit' to exit.")
    print("Using VM model: {}".format(settings.VM_MODEL))
    print("VM Address: {}".format(settings.VM_ADDRESS))
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
            
            # Generate response using VM model
            print("Generating response with VM model...")
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
            print(f"Error in chat loop: {e}")

def main():
    """Main function to run the chat interface with VM model"""
    # Initialize connections
    if not initialize_connections():
        sys.exit(1)
    
    # Start chat loop
    chat_loop()

if __name__ == "__main__":
    main()
