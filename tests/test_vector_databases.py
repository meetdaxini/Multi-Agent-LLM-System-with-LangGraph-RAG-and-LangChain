from my_rag.embeddings.huggingface_embedding import HuggingFaceEmbedding
from my_rag.vectorstores.deeplake_store import DeepLakeVectorStore
import numpy as np

def main():
    # Initialize the embedding model
    embedding_model = HuggingFaceEmbedding(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        device='cpu'
    )

    # Sample documents
    documents = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Deep learning models are very powerful.",
        "Natural language processing enables computers to understand text.",
        "The fox is quick and the dog is lazy."
    ]

    # Generate embeddings for the documents
    document_embeddings = embedding_model.embed(documents)
    document_embeddings = np.array(document_embeddings)

    # Metadata for each document
    metadata = [{'text': doc} for doc in documents]

    # Initialize the vector store
    vector_store = DeepLakeVectorStore(dataset_path='my_vector_store')

    # Add embeddings and metadata to the vector store
    vector_store.add_embeddings(document_embeddings, metadata)

    # Query
    query_text = "What is the role of AI in modern technology?"
    query_embedding = embedding_model.embed([query_text])
    query_embedding = np.array(query_embedding)

    # Search for the top 2 most similar documents
    results = vector_store.search(query_embedding=query_embedding, k=2)

    # Display results
    for idx, result in enumerate(results):
        print(f"Result {idx + 1}:")
        print(f"Text: {result['metadata']['text']}")
        print()

    # Clean up resources
    embedding_model.clean_up()
    vector_store.clean_up()

if __name__ == '__main__':
    main()
