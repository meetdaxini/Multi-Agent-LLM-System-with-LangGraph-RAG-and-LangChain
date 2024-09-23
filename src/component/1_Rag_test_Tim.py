import os
import chromadb
import torch
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate


# Utility Functions
def preprocess_text(text):
    """Function to preprocess text by lowering and stripping whitespace."""
    return text.lower().strip()


def chunk_text(text, tokenizer, chunk_size=512):
    """Splits a text into chunks based on token length."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return [tokenizer.decode(tokens[i:i + chunk_size], skip_special_tokens=True)
            for i in range(0, len(tokens), chunk_size)]


# 1. Embedding Model Class
class EmbeddingModel:
    def __init__(self, model_name='all-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        """Generate embeddings for a list of texts."""
        return self.model.encode(texts, convert_to_tensor=False).tolist()


# 2. Language Model Class
class LLMModel:
    def __init__(self, model_name, access_token):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=access_token)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate_answer(self, prompt, max_length=150):
        """Generates an answer given a prompt."""
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=min(self.tokenizer.model_max_length, 1024),  # Limit max_length
                truncation=True,
                padding='max_length'
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=50,
                    num_return_sequences=1,
                    num_beams=5,
                    early_stopping=True
                )

            return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        except Exception as e:
            print(f"An error occurred during answer generation: {e}")
            return "I'm sorry, but I couldn't generate an answer due to a technical issue."


# 3. Retrieval Model Class
class RetrievalModel:
    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection("collection-documents")

    def add_documents(self, documents, metadatas, ids, embeddings, batch_size=100):
        """Adds documents and their embeddings to ChromaDB."""
        for i in tqdm(range(0, len(documents), batch_size), desc="Adding documents to Chroma"):
            batch_docs = documents[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            try:
                self.collection.add(
                    documents=batch_docs,
                    metadatas=batch_meta,
                    ids=batch_ids,
                    embeddings=batch_embeddings
                )
            except Exception as e:
                print(f"An error occurred while adding batch {i // batch_size}: {e}")

    def query(self, query_embedding, n_results=3):
        """Queries ChromaDB for the top documents based on the query embedding."""
        return self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )


# 4. Generation Model Class
class GenerationModel:
    def __init__(self, llm_model, tokenizer):
        self.llm_model = llm_model
        self.tokenizer = tokenizer

    def generate(self, query, retrieved_docs):
        """Generates an answer based on the query and retrieved documents."""
        # Combine retrieved documents into a single context
        context = " ".join(retrieved_docs)

        # Split context into smaller chunks if too long
        max_input_length = min(self.tokenizer.model_max_length - 50, 512)
        context_chunks = []
        context_tokens = self.tokenizer.encode(context, add_special_tokens=False)

        for i in range(0, len(context_tokens), max_input_length):
            chunk = context_tokens[i:i + max_input_length]
            context_chunks.append(self.tokenizer.decode(chunk, skip_special_tokens=True))

        # Use only the first chunk if the context is too long
        if context_chunks:
            context = context_chunks[0]
        else:
            context = context[:max_input_length]

        # Create a prompt for the LLM
        prompt = f"""
        Please provide a concise answer to the question below, using the provided context in 75 words.

        Context:
        {context}

        Question:
        {query}

        Answer:
        """

        return self.llm_model.generate_answer(prompt)


# 5. RAG System Class
class RAGSystem:
    def __init__(self, embedding_model, retrieval_model, llm_model, generation_model):
        self.embedding_model = embedding_model
        self.retrieval_model = retrieval_model
        self.llm_model = llm_model
        self.generation_model = generation_model

    def add_documents(self, dataset):
        """Processes and adds documents to the retrieval model."""
        documents, metadatas, ids = [], [], []
        for i, row in enumerate(dataset):
            # Preprocess and chunk the text
            chunks = chunk_text(f"Question: {row['question']} Context: {row['context']}",
                                self.llm_model.tokenizer)
            documents.extend(chunks)
            metadatas.extend([{"question_id": row['question_id']} for _ in chunks])
            ids.extend([f"doc_{i}_chunk_{j}" for j in range(len(chunks))])

        # Generate embeddings
        embeddings = self.embedding_model.encode(documents)

        # Add documents to Chroma
        self.retrieval_model.add_documents(documents, metadatas, ids, embeddings)

    def query(self, query, n_results=3):
        """Handles the full RAG query workflow: retrieval and generation."""
        # Preprocess and embed the query
        processed_query = preprocess_text(query)
        query_embedding = self.embedding_model.encode([processed_query])

        # Retrieve top documents
        results = self.retrieval_model.query(query_embedding, n_results=n_results)
        if not results['documents'] or not results['documents'][0]:
            return "I don't have enough information to answer this question."

        retrieved_docs = results['documents'][0]
        # print(f"Retrieved Documents: {retrieved_docs}")  # Debugging print

        # Generate the response based on retrieved documents
        response = self.generation_model.generate(query, retrieved_docs)
        #  print(f"Generated Response: {response}")  # Debugging print
        return response


# Setup and Initialize RAG System
def setup_rag_system():
    """Initializes the RAG system with models and dataset."""
    # Set Hugging Face API key and load models
    access_token_huggingface = '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'
    embedding_model = EmbeddingModel(model_name='all-mpnet-base-v2')
    llm_model = LLMModel(model_name="t5-large", access_token=access_token_huggingface)
    retrieval_model = RetrievalModel()
    generation_model = GenerationModel(llm_model, llm_model.tokenizer)
    rag_system = RAGSystem(embedding_model, retrieval_model, llm_model, generation_model)

    # Load dataset and add documents
    dataset = load_dataset("explodinggradients/ragas-wikiqa")['train']
    rag_system.add_documents(dataset)

    return rag_system


def interactive_query(rag_system):
    """Starts an interactive query loop for the RAG system."""
    print("RAG System is ready! You can start asking questions.")
    print("Type 'exit' to end the session.")
    while True:
        query = input("Enter your question: ")
        if query.lower() == 'exit':
            print("Exiting the session. Goodbye!")
            break

        response = rag_system.query(query)
        print(f"Response: {response}")
        print('-' * 10, '\n')


#%%
if __name__ == "__main__":
    # Setup RAG system
    rag_system = setup_rag_system()

    # Start interactive query loop
    interactive_query(rag_system)
