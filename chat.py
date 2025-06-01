import sys
from typing import List, Dict
from openai import OpenAI
from pymilvus import connections, Collection
from config import settings

class LocalPDFChat:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        
        # Connect to Milvus
        connections.connect("default", host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)
        
        # Load collection
        try:
            self.collection = Collection(settings.COLLECTION_NAME)
            self.collection.load()
            print(f"✅ Connected to collection '{settings.COLLECTION_NAME}'")
        except Exception as e:
            print(f"❌ Error connecting to collection: {e}")
            print("Please run the vectorizer first to create the collection")
            sys.exit(1)
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for user query."""
        response = self.client.embeddings.create(
            input=[query],
            model="text-embedding-3-large"
        )
        return response.data[0].embedding
    
    def search_similar_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar documents in Milvus."""
        query_embedding = self.generate_query_embedding(query)
        
        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 10}
        }
        
        results = self.collection.search(
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
    
    def generate_response(self, query: str, context_docs: List[Dict]) -> str:
        """Generate response using OpenAI with context from documents."""
        # Prepare context
        context_text = "\n\n".join([
            f"From {doc['file_name']} (page {doc['page_number']}):\n{doc['text']}"
            for doc in context_docs
        ])
        
        # Create prompt
        prompt = f"""You are a helpful AI assistant that answers questions based on PDF documents.
Use the following context from the documents to answer the user's question. If the answer is not in the context, say so.

Context:
{context_text}

Question: {query}

Answer:"""

        response = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided document context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    def chat_loop(self):
        """Main chat loop."""
        print("\n🤖 PDF Chat Assistant")
        print("=" * 50)
        print("Ask questions about your PDF documents. Type 'quit' to exit.")
        print()
        
        while True:
            try:
                query = input("You: ").strip()
                
                if query.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye!")
                    break
                
                if not query:
                    continue
                
                print("🔍 Searching for relevant documents...")
                
                # Search for similar documents
                similar_docs = self.search_similar_documents(query, top_k=3)
                
                if not similar_docs:
                    print("❌ No relevant documents found for your query.")
                    continue
                
                print(f"📚 Found {len(similar_docs)} relevant document(s)")
                
                # Generate response
                print("🤖 Generating response...")
                response = self.generate_response(query, similar_docs)
                
                print(f"\nAssistant: {response}\n")
                
                # Show sources
                print("📖 Sources:")
                for i, doc in enumerate(similar_docs, 1):
                    print(f"  {i}. {doc['file_name']} (page {doc['page_number']}) - Score: {doc['score']:.3f}")
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main function to run the chat interface."""
    chat = LocalPDFChat()
    chat.chat_loop()

if __name__ == "__main__":
    main()
