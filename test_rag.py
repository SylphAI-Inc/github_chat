import os
from rag import RAG
import adalflow as adal
from adalflow.core.db import LocalDB
from adalflow.core.types import Document
from adalflow.utils import get_adalflow_default_root_path
from adalflow.components.data_process import TextSplitter, ToEmbeddings
import tempfile

def create_sample_documents():
    """Create some sample documents for testing."""
    sample_texts = [
        """Alice is a software engineer who loves coding in Python. 
        She specializes in machine learning and has worked on several NLP projects.
        Her favorite project was building a chatbot for customer service.""",
        
        """Bob is a data scientist with expertise in deep learning.
        He has published papers on transformer architectures and attention mechanisms.
        Recently, he's been working on improving RAG systems.""",
        
        """The company cafeteria serves amazing tacos on Tuesdays.
        They also have a great coffee machine that makes perfect lattes.
        Many employees enjoy their lunch breaks in the outdoor seating area."""
    ]
    
    return [Document(text=text, meta_data={"title": f"doc_{i}"}) 
            for i, text in enumerate(sample_texts)]

def initialize_test_database(db_path: str):
    """Initialize database with sample documents."""
    documents = create_sample_documents()
    
    data_transformer = adal.Sequential(
        TextSplitter(split_by="word", chunk_size=50, chunk_overlap=20),
        ToEmbeddings(
            embedder=adal.Embedder(
                model_client=adal.OpenAIClient(),
                model_kwargs={
                    "model": "text-embedding-3-small",
                    "dimensions": 256,
                    "encoding_format": "float",
                }
            ),
            batch_size=100
        )
    )
    
    db = LocalDB("test_db")
    db.register_transformer(transformer=data_transformer, key="split_and_embed")
    db.load(documents)
    db.transform(key="split_and_embed")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db.save_state(filepath=db_path)
    return db_path

def main():
    # Set your OpenAI API key
    os.environ["OPENAI_API_KEY"] = "Your OpenAI API key"
    
    # Create a temporary directory for the database
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_db")
    
    # Initialize test database
    initialize_test_database(db_path)
    
    # Create RAG instance
    rag = RAG(index_path=db_path)
    
    # Test conversation flow with memory
    test_conversation = [
        "Who is Alice and what does she do?",
        "What about Bob? What's his expertise?",
        "Can you tell me more about what the previous person works on?",
        "What was her favorite project?",  # Tests memory of Alice
        "Between these two people, who has more experience with RAG systems?",  # Tests memory of both
        "Do they ever meet? Where might they have lunch together?"  # Tests memory and context combination
    ]
    
    print("Starting conversation test with memory...\n")
    for i, query in enumerate(test_conversation, 1):
        print(f"\n----- Query {i} -----")
        print(f"User: {query}")
        try:
            # Get conversation history before the response
            print("\nCurrent Conversation History:")
            history = rag.memory()
            if history:
                print(history)
            else:
                print("(No history yet)")
                
            response, docs = rag(query)
            print(f"\nAssistant: {response}")
            
            # Show most relevant document used
            if docs:
                most_relevant = docs[0].documents[0].text.strip()
                print(f"\nMost relevant context used: \n{most_relevant[:200]}...")
                
        except Exception as e:
            print(f"Error: {e}")
        print("\n" + "="*50)

if __name__ == "__main__":
    main() 