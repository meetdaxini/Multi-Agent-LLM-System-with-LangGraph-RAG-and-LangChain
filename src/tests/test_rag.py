from my_rag.components.embeddings.huggingface_embedding import HuggingFaceEmbedding
from my_rag.components.vectorstores.deeplake_store import DeepLakeVectorStore
from my_rag.components.llms.huggingface_llm import HuggingFaceLLM
import numpy as np
import torch

def main():
    embedding_model = HuggingFaceEmbedding(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        device='cpu'
    )
    vector_store = DeepLakeVectorStore(dataset_path='my_vector_store')
    llm = HuggingFaceLLM(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        device='cuda' if torch.cuda.is_available() else 'cpu',
        torch_dtype=torch.float16,
        load_in_8bit=True,
        device_map='auto',
        trust_remote_code=True,
    )
    documents = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Deep learning models are very powerful.",
        "Natural language processing enables computers to understand text.",
        "The fox is quick and the dog is lazy."
        "Quantum computers work differently from classical computers in a number of ways, including: Qubits Superposition Quantum entanglement",
        "A quantum computer is a computer that exploits quantum mechanical phenomena. On small scales, physical matter exhibits properties of both particles and waves, and quantum computing leverages this behavior using specialized hardware.",
        "Quantum computers were developed by the team of two scientists: Meet and Tim"
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

    # Sample question
    question = "How does quantum computing work and Who created Quantum Computers?"
    # Generate query embedding
    query_embedding = embedding_model.embed([question])
    query_embedding = np.array(query_embedding)

    # Retrieve relevant documents
    results = vector_store.search(query_embedding=query_embedding, k=3)
    context = " ".join([res['metadata']['text'] for res in results])

    # Generate answer using LLM with context
    answer = llm.generate_response_with_context(
        context=context,
        prompt=question,
        max_length=256,
        temperature=0.7,
        top_p=0.9,
    )

    print("Question:")
    print(question)
    print("\nAnswer:")
    print(answer)

    # Clean up resources
    embedding_model.clean_up()
    vector_store.clean_up()
    llm.clean_up()

if __name__ == '__main__':
    main()
