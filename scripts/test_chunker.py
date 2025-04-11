from raglab.chunker import Chunker

if __name__ == "__main__":
    file_path = "data/meditations_marcus_aurelius.txt"
    chunker = Chunker()

    chunks = chunker.chunk_file(file_path, max_tokens=200, overlap=20)

    print(f"\n✅ Chunks created: {len(chunks)}\n")
    print("🔍 First chunk:\n")
    print(chunks[0])
