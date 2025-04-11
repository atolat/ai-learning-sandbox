from pathlib import Path
from raglab.chunker import Chunker
from raglab.embedding import Embedder
from raglab.vector_store import VectorStore
from raglab.rag_engine import RAGEngine

def test_chunker_and_embedding():
    # Initialize components
    chunker = Chunker()
    embedder = Embedder()
    
    # Test text
    test_text = """
    The only way to do great work is to love what you do.
    If you haven't found it yet, keep looking. Don't settle.
    As with all matters of the heart, you'll know when you find it.
    """
    
    # Test chunking
    print("\nTesting Chunker:")
    chunks = chunker.chunk_text(test_text, max_tokens=20, overlap=5)
    print(f"Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(chunk)
    
    # Test embeddings
    print("\nTesting Embedder:")
    embeddings = embedder.embed(chunks)
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {len(embeddings[0])}")
    
    # Test with file
    print("\nTesting with file:")
    file_path = Path("data/meditations_marcus_aurelius.txt")
    if file_path.exists():
        file_chunks = chunker.chunk_file(str(file_path), max_tokens=100, overlap=20)
        print(f"Number of chunks from file: {len(file_chunks)}")
        file_embeddings = embedder.embed(file_chunks)
        print(f"Number of embeddings from file: {len(file_embeddings)}")
    else:
        print("Meditations file not found")

def test_vector_store():
    print("\nTesting Vector Store:")
    
    # Initialize components
    embedder = Embedder()
    vector_store = VectorStore()
    
    # Test documents
    documents = [
        "The quick brown fox jumps over the lazy dog.",
        "A fox is a clever and agile animal.",
        "Dogs are known for their loyalty and companionship.",
        "The weather is beautiful today.",
        "Machine learning is transforming technology."
    ]
    
    # Generate embeddings and add to store
    print("\nAdding documents to vector store:")
    embeddings = embedder.embed(documents)
    for text, embedding in zip(documents, embeddings):
        vector_store.add(text, embedding)
    print(f"Added {len(documents)} documents")
    
    # Test search
    print("\nTesting search:")
    query = "Tell me about foxes"
    query_embedding = embedder.embed([query])[0]
    
    results = vector_store.search(query_embedding, k=2)
    print(f"\nQuery: '{query}'")
    print("\nTop 2 most similar documents:")
    for i, (text, score) in enumerate(results, 1):
        print(f"\n{i}. Score: {score:.4f}")
        print(f"Text: {text}")

def test_end_to_end_rag():
    print("\nTesting End-to-End RAG:")
    
    # Initialize components
    chunker = Chunker()
    embedder = Embedder()
    vector_store = VectorStore()
    
    # Load and process the Meditations text
    file_path = Path("data/meditations_marcus_aurelius.txt")
    if not file_path.exists():
        print("Meditations file not found")
        return
        
    print(f"\nProcessing file: {file_path}")
    
    # Chunk the text
    chunks = chunker.chunk_file(str(file_path), max_tokens=100, overlap=20)
    print(f"Created {len(chunks)} chunks")
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings = embedder.embed(chunks)
    print(f"Generated {len(embeddings)} embeddings")
    
    # Add to vector store
    print("Adding to vector store...")
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vector_store.add(chunk, embedding)
        if (i+1) % 10 == 0:
            print(f"Added {i+1}/{len(chunks)} chunks")
    print(f"Added all {len(chunks)} chunks to vector store")
    
    # Initialize RAG engine
    rag_engine = RAGEngine(embedder, vector_store)
    
    # Test queries
    queries = [
        "What does Marcus Aurelius say about death?",
        "How should one deal with anger?",
        "What is the importance of reason?",
        "How to live a good life according to Marcus Aurelius?"
    ]
    
    print("\nTesting semantic search with queries:")
    for query in queries:
        print(f"\nQuery: '{query}'")
        
        # First, show the raw retrieval results
        query_embedding = embedder.embed([query])[0]
        results = vector_store.search(query_embedding, k=3)
        print("\nTop 3 most relevant passages:")
        for i, (text, score) in enumerate(results, 1):
            print(f"\n{i}. Score: {score:.4f}")
            print(f"Text: {text[:200]}..." if len(text) > 200 else f"Text: {text}")
        
        # Then, use the RAG engine to generate an answer
        print("\nRAG Engine Answer:")
        answer = rag_engine.query(query)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    test_chunker_and_embedding()
    test_vector_store()
    test_end_to_end_rag() 