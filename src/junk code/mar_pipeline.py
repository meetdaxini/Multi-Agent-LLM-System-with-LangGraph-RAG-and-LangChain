from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from my_rag.components.memory_db.improved_mar import ChromaMemoryDB
from my_rag.components.llms.huggingface_llm import HuggingFaceLLM
from my_rag.components.embeddings.huggingface_embedding import HuggingFaceEmbedding
from my_rag.components.memory_db.improved_mar import process_query
from typing import Dict, List
import pandas as pd
import chromadb
import logging
import torch
import os
import gc


DEFAULT_DATA_PATH = "data_test"  # Path to your PDFs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# class Question:
#     def __init__(self, text, embedding_model_path, chromadb_handler, llm, k=2, chunk_size=1000, device=None):
#         """
#         Initialize a Question object.
#
#         Parameters:
#             text (str): The question text.
#             embedding_model_path (str): Path or model name for HuggingFaceEmbedding.
#             chromadb_handler (ChromaDBHandler): Handler for querying the vector DB.
#             llm (HuggingFaceLLM): An instance of HuggingFaceLLM for generating answers.
#             k (int): Number of relevant documents to retrieve.
#             chunk_size (int): Number of tokens per chunk.
#             device (Optional[str]): The device to load the embedding model on (e.g., 'cpu', 'cuda').
#         """
#         self.text = text
#         self.embedder = HuggingFaceEmbedding(model_name=embedding_model_path, device=device)  # Initialize embedder
#         self.chromadb_handler = chromadb_handler
#         self.llm = llm
#         self.k = k
#         self.chunk_size = chunk_size
#         self.embedding = None
#         self.retrieved_documents = []
#         self.answer = None
#
#     def embed_question(self):
#         """Generate embeddings for the question text."""
#         try:
#             # Generate and store the embedding
#             self.embedding = self.embedder.embed([self.text])[0].cpu().numpy()
#             logger.info(f"Question embedding generated with shape: {self.embedding.shape}")
#         except Exception as e:
#             logger.error(f"Failed to embed question: {e}")
#             raise
#
#     def retrieve_documents(self):
#         """Query the vector DB to retrieve relevant documents."""
#         if self.embedding is None:
#             raise ValueError("Question embedding not generated. Call 'embed_question' first.")
#
#         logger.info(f"Querying ChromaDB with embedding shape: {self.embedding.shape}")
#
#         # Attempt to retrieve from the vector DB
#         results = self.chromadb_handler.query(query_embeddings=[self.embedding], n_results=self.k)
#
#         # Ensure results structure is valid
#         if not results or "documents" not in results or not results["documents"][0]:
#             logger.warning("No documents found in ChromaDB for the query.")
#             raise ValueError("No documents retrieved. Ensure the database is populated and embeddings are correct.")
#
#         # Store retrieved documents
#         self.retrieved_documents = results["documents"][0]
#         logger.info(f"Retrieved {len(self.retrieved_documents)} documents for the query.")
#         return self.retrieved_documents
#
#     def generate_answer(self, dataframe, context_field="context"):
#         """
#         Generate an answer using the LLM's internal prompt format, with retrieved documents as context.
#
#         Parameters:
#             dataframe (pd.DataFrame): DataFrame containing retrieved documents and context text.
#             context_field (str): Column name for the context text in the DataFrame.
#         """
#         if dataframe.empty:
#             raise ValueError("DataFrame is empty. No context available to generate an answer.")
#
#         # Combine relevant context texts from retrieved documents
#         context = "\n\n".join(dataframe[context_field].tolist())
#         prompt = self.text  # Original question text as prompt
#
#         try:
#             # Generate response using the provided context
#             response = self.llm.generate_response_with_context(
#                 context=context, max_length=200, temperature=0.7, top_p=0.9, prompt=prompt
#             )
#             self.answer = response.strip()
#             logger.info("Answer generated successfully.")
#         except Exception as e:
#             logger.error(f"Error generating answer: {e}")
#             self.answer = "Could not generate a relevant answer."
#
#         return self.answer


def load_dataset(dataset_path=DEFAULT_DATA_PATH):
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path '{dataset_path}' does not exist.")

    document_loader = PyPDFDirectoryLoader(dataset_path)
    documents = document_loader.load()

    chunked_texts, document_ids = [], []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=115)

    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        if chunks:
            chunked_texts.extend(chunks)
            document_ids.extend([doc.metadata.get("source", "")] * len(chunks))
        else:
            logger.warning(f"No chunks created for document: {doc.metadata.get('source', '')}")

    df = pd.DataFrame({
        "context": chunked_texts,
        "source_doc": document_ids,
    })

    if df.empty:
        raise ValueError("No data found in data_test directory.")

    return df


class ChromaDBHandler:
    def __init__(self, collection_name):
        self.collection_name = collection_name
        self.client = chromadb.Client()  # Use in-memory client

        try:
            self.client.delete_collection(name=collection_name)
        except Exception as e:
            logger.info(f"No previous collection to delete. Starting fresh: {e}")

        self.collection = self.client.create_collection(name=collection_name)
        logger.info(f"Created ChromaDB collection (database) with name: {self.collection_name}")

        # Track the document count manually
        self.document_count = 0

    def add_embeddings(self, embeddings, documents, metadatas, ids):
        if embeddings is None or len(embeddings) == 0:
            raise ValueError("No embeddings to add to ChromaDB.")

        if not (len(embeddings) == len(documents) == len(metadatas) == len(ids)):
            raise ValueError("Embeddings, documents, metadatas, and ids must all have the same length.")

        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
        logger.info(f"Added {len(embeddings)} embeddings to the ChromaDB collection: {self.collection_name}")

        # Update the document count
        self.document_count += len(embeddings)

    def query(self, query_embeddings, n_results, include=None):
        if include is None:
            include = ["documents", "metadatas"]
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            include=include,
        )
        return results

    def delete_collection(self):
        self.client.delete_collection(name=self.collection_name)
        self.document_count = 0

    def verify_population(self):
        logger.info(f"Collection '{self.collection_name}' has {self.document_count} documents.")
        if self.document_count == 0:
            raise RuntimeError("ChromaDB collection is empty. Embeddings were not added successfully.")

    def save_to_chromadb(self, embeddings, documents, metadatas, ids):
        """Saves embeddings and associated data to ChromaDB."""
        self.add_embeddings(embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids)

    def retrieve_from_chromadb(self, query_embedding, n_results):
        """Retrieves relevant documents using ChromaDBHandler."""
        results = self.query(query_embeddings=[query_embedding], n_results=n_results)

        # Ensure results has a consistent structure
        if not results or "documents" not in results or not results["documents"]:
            return {"documents": [], "distances": []}  # Return empty structure if no results

        return results


class ImprovedMemoryAugmentedRAG:
    def __init__(
            self,
            embedding_model,
            embedding_dim: int,
            llm,
            chromadb_handler: ChromaDBHandler,  # Move this up
            memory_db: ChromaMemoryDB,  # Move this up
            collection_name: str = "memory_store",
            max_memory_size: int = 10000,
            similarity_threshold: float = 0.75,
            cache_size: int = 1000
    ):
        """
        Initialize the Memory Augmented RAG system.

        Args:
            embedding_model: Model to generate embeddings
            embedding_dim: Dimension of the embeddings
            llm: Language model for generating responses
            chromadb_handler: Handler for main ChromaDB collection
            memory_db: In-memory database for frequently accessed documents
            collection_name: Name for the ChromaDB collection
            max_memory_size: Maximum number of documents to store
            similarity_threshold: Threshold for similarity search
            cache_size: Size of the cache for frequent queries
        """
        self.embedder = embedding_model
        self.llm = llm
        self.chromadb_handler = chromadb_handler
        self.memory_db = memory_db

        # Initialize ChromaMemoryDB with correct parameters
        self.memory_db = ChromaMemoryDB(
            collection_name=collection_name,
            embedding_dim=embedding_dim,
            similarity_threshold=similarity_threshold,
            cache_size=cache_size,
            max_memory_size=max_memory_size
        )

    def process_query(self, query_text: str, k: int = 5) -> Dict:
        logger.info(f"Processing query with k={k}")

        # Generate query embedding
        query_embedding = self.embedder.embed([query_text])[0].clone().detach().to(torch.float32)

        # Step 1: Attempt to retrieve relevant documents from MemoryDB
        retrieved_docs = self.memory_db.retrieve_top_k(query_embedding, k=k)

        # Step 2: If MemoryDB has no relevant documents, proceed to query ChromaDB
        if not retrieved_docs:
            logger.info("No similar document found in memory. Falling back to main collection.")
            fallback_results = self.chromadb_handler.retrieve_from_chromadb(query_embedding.numpy(), n_results=k)

            if (
                    fallback_results and
                    fallback_results.get("documents") and
                    fallback_results.get("distances")
            ):
                retrieved_docs = [
                    (doc, 1 - distance)  # Convert distance to similarity score
                    for doc, distance in zip(fallback_results["documents"][0], fallback_results["distances"][0])
                ]
            else:
                logger.warning("No similar documents found in main collection.")
                retrieved_docs = []

        # Step 3: Prepare response
        if retrieved_docs:
            context = "\n\n".join([doc for doc, _ in retrieved_docs])
            prompt = (
                "Use the following context to answer the question. If the answer "
                "cannot be found in the context, say 'I don't have enough information "
                "to answer this question.'\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query_text}\n\n"
                "Answer:"
            )
            response = self.llm.generate(prompt)
            return {
                "answer": response,
                "retrieved_documents": [doc for doc, _ in retrieved_docs],
                "scores": [score for _, score in retrieved_docs]
            }
        else:
            return {
                "answer": "I don't have enough information to answer this question.",
                "retrieved_documents": [],
                "scores": []
            }

    def add_to_memory(self, documents: List[str]):
        """
        Add new documents to memory.

        Args:
            documents: List of documents to add to memory
        """
        for doc in documents:
            # Generate embedding and convert to torch tensor
            embedding = torch.tensor(
                self.embedder.embed([doc])[0],
                dtype=torch.float32
            )
            # Add to ChromaMemoryDB
            self.memory_db.add_to_memory(embedding, doc)
            logger.info(f"Added document to memory: {doc[:100]}...")


class MARPipeline:
    def __init__(self):
        self.memory_db = chromadb.connect("path_to_memory_db")
        self.main_db = chromadb.connect("path_to_main_db")

    def query_pipeline(self, query):
        # Step 1: Process the query with memory-first approach
        result = process_query(query)
        # Step 2: Generate response based on retrieved info
        generated_response = self.generate_response(result)
        return generated_response

    def generate_response(self, retrieved_data):
        # Use LLM or RAG-based model to generate response
        return f"Generated answer based on: {retrieved_data}"


class Question:
    def __init__(self, text, rag_system: ImprovedMemoryAugmentedRAG, k=2):
        """
        Initialize a Question object.

        Parameters:
            text (str): The question text.
            rag_system (ImprovedMemoryAugmentedRAG): An instance of the RAG system to manage retrieval and LLM.
            k (int): Number of relevant documents to retrieve.
        """
        self.text = text
        self.rag_system = rag_system
        self.k = k
        self.answer = None
        self.retrieved_documents = []

    def retrieve_documents(self):
        """Retrieve relevant documents using RAG system's memory-augmented query process."""
        if self.k <= 0:
            raise ValueError("The number of documents to retrieve (k) must be greater than zero.")

        result = self.rag_system.process_query(self.text, k=self.k)
        self.retrieved_documents = result["retrieved_documents"]
        return result

    def generate_answer(self):
        """Generate an answer using the retrieved documents."""
        if not self.retrieved_documents:
            raise ValueError("No documents retrieved. Call 'retrieve_documents' first.")

        context = "\n\n".join(self.retrieved_documents)
        prompt = self.text
        self.answer = self.rag_system.llm.generate(
            prompt=f"Context:\n{context}\n\nQuestion: {prompt}\nAnswer:"
        ).strip()

        return self.answer


def log_cuda_memory_usage(message=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1536 ** 2)
        reserved = torch.cuda.memory_reserved() / (1536 ** 2)
        logger.info(f"{message} - CUDA memory allocated: {allocated:.2f} MB, reserved: {reserved:.2f} MB")
    else:
        logger.info(f"{message} - CUDA not available")


def create_document_embeddings(
        embedding_model,
        dataframe,
        context_field="context",
        doc_id_field="source_doc",
        batch_size=32,
        chunk_size=1000,
        chunk_overlap=115,
        embed_document_method="embed_documents",
        instruction="",
        max_length=None,
):
    contexts = dataframe[context_field].tolist()
    document_ids = dataframe[doc_id_field].tolist()

    chunked_texts, chunked_doc_ids = [], []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    for context, doc_id in zip(contexts, document_ids):
        chunks = text_splitter.split_text(context)
        if chunks:
            chunked_texts.extend(chunks)
            chunked_doc_ids.extend([doc_id] * len(chunks))
        else:
            logger.warning(f"No chunks created for document ID {doc_id}.")

    if not chunked_texts:
        raise ValueError("No text chunks found to create embeddings. Check document loading and splitting.")

    log_cuda_memory_usage("Before creating context embeddings")

    if hasattr(embedding_model, embed_document_method):
        embed_func = getattr(embedding_model, embed_document_method)
        embeddings = embed_func([f"{instruction}{text}" for text in chunked_texts] if instruction else chunked_texts)
    else:
        embeddings = embedding_model.embed(
            chunked_texts, batch_size=batch_size, instruction=instruction, max_length=max_length
        )

    if embeddings is None or len(embeddings) == 0:
        raise RuntimeError("Embedding function returned an empty result.")

    embeddings = embeddings.cpu().numpy()
    torch.cuda.empty_cache()
    gc.collect()
    log_cuda_memory_usage("After processing context embeddings")

    return chunked_texts, chunked_doc_ids, embeddings


def main():
    # Initialize embedding model
    embedding_model_path = "dunzhang/stella_en_1.5B_v5"
    embedding_model = HuggingFaceEmbedding(
        model_name=embedding_model_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Initialize LLM
    llm_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    llm = HuggingFaceLLM(
        model_name=llm_model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float16,
        load_in_8bit=True,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load dataset and prepare ChromaDB collection
    df = load_dataset()
    chromadb_handler = ChromaDBHandler(collection_name="main_collection")
    chunked_texts, document_ids = df["context"].tolist(), df["source_doc"].tolist()

    # Populate ChromaDB if empty
    if chromadb_handler.document_count == 0:
        embeddings = embedding_model.embed(chunked_texts).detach().cpu().numpy()
        chromadb_handler.save_to_chromadb(
            embeddings=embeddings,
            documents=chunked_texts,
            metadatas=[{"source": doc_id} for doc_id in document_ids],
            ids=[str(i) for i in range(len(chunked_texts))]
        )

    # Initialize memory database
    memory_db = ChromaMemoryDB(
        collection_name="interaction_memory",
        embedding_dim=embedding_model.embed(["Sample text"])[0].shape[0],  # Set embedding dimension dynamically
        similarity_threshold=0.75,
        cache_size=1000,
        max_memory_size=10000
    )

    # Initialize ImprovedMemoryAugmentedRAG
    rag_system = ImprovedMemoryAugmentedRAG(
        embedding_model=embedding_model,
        embedding_dim=embedding_model.embed(["Sample text"])[0].shape[0],
        llm=llm,
        chromadb_handler=chromadb_handler,
        memory_db=memory_db
    )

    # Prepopulate memory with frequently asked questions or common documents
    prepopulated_documents = [
        "Question: Is Hirschsprung disease classified as a single-gene disorder or influenced by multiple factors? Answer: Coding sequence mutations in RET, GDNF, EDNRB, EDN3, and SOX10 are involved in the development of Hirschsprung disease. The majority of these genes was shown to be related to Mendelian syndromic forms of Hirschsprung's disease, whereas the non-Mendelian inheritance of sporadic non-syndromic Hirschsprung disease proved to be complex; involvement of multiple loci was demonstrated in a multiplicative model.",
        "Question: List EGFR receptor signaling molecules. Answer: The 7 known EGFR ligands are: epidermal growth factor (EGF), betacellulin (BTC), epiregulin (EPR), heparin-binding EGF (HB-EGF), transforming growth factor-α [TGF-α], amphiregulin (AREG), and epigen (EPG).",
        "Question: Do long non-coding RNAs get spliced? Answer: Long non coding RNAs appear to be spliced through the same pathway as the mRNAs."
    ]

    rag_system.add_to_memory(prepopulated_documents)

    # Initialize a Question instance
    question_text = "Is Hirschsprung disease classified as a single-gene disorder or influenced by multiple factors?"
    question_instance = Question(text=question_text, rag_system=rag_system, k=3)

    # Retrieve relevant documents
    result = question_instance.retrieve_documents()
    retrieved_docs = result.get("retrieved_documents", [])

    # Check if documents were successfully retrieved
    if not retrieved_docs:
        logger.error("No documents retrieved. Check memory and ChromaDB population.")
        print("No documents retrieved. Exiting.")
        return

    # Generate answer only if documents are found
    answer = question_instance.generate_answer()
    print(f"\n** Question: {question_text}")
    print(f"** Answer: {answer}")
    print("\n** Retrieved Documents:")
    for doc in retrieved_docs:
        print(f"** Document: {doc}\n")


if __name__ == "__main__":
    main()



