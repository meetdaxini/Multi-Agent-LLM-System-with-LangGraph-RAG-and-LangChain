import pandas as pd
import torch
import logging
import gc
from langchain.text_splitter import RecursiveCharacterTextSplitter
from my_rag_ollama.get_embedding_function import (
    get_msmarco_embeddings,
)
from my_rag.components.embeddings.huggingface_embedding import HuggingFaceEmbedding
import chromadb
import re
import os
from langchain_community.document_loaders import PyPDFLoader


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def alphanumeric_string(input_string):
    return re.sub(r"[^a-zA-Z0-9]", "", input_string)


class ChromaDBHandler:
    """
    A handler class for ChromaDB operations.
    """

    def __init__(self, collection_name, host="localhost", port=8000):
        """
        Initializes the ChromaDB client and creates a collection.

        Args:
            collection_name (str): Name of the collection.
            host (str): Host of the ChromaDB server.
            port (int): Port of the ChromaDB server.
        """
        self.collection_name = collection_name
        self.client = chromadb.HttpClient(host=host, port=port)
        # Delete existing collection with the same name
        try:
            self.client.delete_collection(name=collection_name)
        except Exception:
            pass
        self.collection = self.client.create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add_embeddings(self, embeddings, documents, metadatas, ids):
        """
        Adds embeddings to the collection.

        Args:
            embeddings (List[List[float]]): Embeddings to add.
            documents (List[str]): List of documents.
            metadatas (List[Dict]): List of metadatas.
            ids (List[str]): List of document IDs.
        """
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

    def query(self, query_embeddings, n_results, include=["metadatas"]):
        """
        Queries the collection.

        Args:
            query_embeddings (List[List[float]]): Query embeddings.
            n_results (int): Number of results to return.
            include (List[str]): What to include in the results.

        Returns:
            Dict: Query results.
        """
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            include=include,
        )
        return results

    def delete_collection(self):
        """
        Deletes the collection.
        """
        self.client.delete_collection(name=self.collection_name)


def log_cuda_memory_usage(message=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        logger.info(
            f"{message} - CUDA memory allocated: {allocated:.2f} MB, reserved: {reserved:.2f} MB"
        )
    else:
        logger.info(f"{message} - CUDA not available")


def load_pdf(pdf_path: str) -> str:
    """
    Loads a PDF and returns its text content.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        str: Text content of the PDF
    """
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    return "\n".join(page.page_content for page in pages)


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Loads the dataset from CSV and associated PDFs.

    Args:
        csv_path (str): Path to the CSV file
        pdf_dir (str): Directory to store/read PDFs

    Returns:
        pd.DataFrame: Loaded dataset with context from PDFs
    """
    # Create PDF directory if it doesn't exist
    os.makedirs(
        "/home/ubuntu/Multi-Agent-LLM-System-with-LangGraph-RAG-and-LangChain/filtered_dataset_csv_pdfs/",
        exist_ok=True,
    )

    # Load CSV
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset="Question ID", keep="first", ignore_index=True)
    contexts = []

    for idx, row in df.iterrows():
        pdf_filename = row["PDF Reference"]
        pdf_path = os.path.join(
            "/home/ubuntu/Multi-Agent-LLM-System-with-LangGraph-RAG-and-LangChain/filtered_dataset_csv_pdfs/",
            pdf_filename,
        )
        try:
            context = load_pdf(pdf_path)
            contexts.append(context)
        except Exception as e:
            logger.error(f"Failed to load PDF for row {idx}: {e}")
            contexts.append("")

    df["context"] = contexts
    return df


def create_document_embeddings(
    embedding_model,
    dataframe,
    context_field="context",
    doc_id_field="source_doc",
    batch_size=32,
    embed_document_method="embed_documents",
    instruction="",
    max_length=None,
    chunk_size=2000,
    chunk_overlap=250,
):
    """
    Creates embeddings for the documents, with improved chunking.

    Args:
        embedding_model: The embedding model to use.
        dataframe (pd.DataFrame): The dataset.
        context_field (str): Name of the context field.
        doc_id_field (str): Name of the document ID field.
        batch_size (int): Batch size for embedding contexts.
        embed_document_method (str): The method name to embed documents.
        instruction (str): Instruction prefix for embedding.
        chunk_size (int): The maximum length of each chunk (in words).
        chunk_overlap (int): The number of overlapping words between chunks.

    Returns:
        Tuple[List[str], List, np.ndarray]: chunked_texts, chunked_doc_ids, embeddings (as numpy array).
    """
    contexts = dataframe[context_field].tolist()
    document_ids = dataframe[doc_id_field].tolist()

    chunked_texts = []
    chunked_doc_ids = []

    for context, doc_id in zip(contexts, document_ids):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = text_splitter.split_text(context)

        chunked_texts.extend(chunks)
        chunked_doc_ids.extend([doc_id] * len(chunks))

    log_cuda_memory_usage("Before creating context embeddings")

    # Use the appropriate method to embed documents
    if hasattr(embedding_model, embed_document_method):
        embed_func = getattr(embedding_model, embed_document_method)
        if instruction:
            instruction_pairs = [f"{instruction}{text}" for text in chunked_texts]
            embeddings = embed_func(instruction_pairs)
        else:
            embeddings = embed_func(chunked_texts)
    else:
        embeddings = embedding_model.embed(
            chunked_texts,
            batch_size=batch_size,
            instruction=instruction,
            max_length=max_length,
        )
        embeddings = embeddings.cpu().numpy()

    torch.cuda.empty_cache()
    gc.collect()
    log_cuda_memory_usage("After processing context embeddings")

    return chunked_texts, chunked_doc_ids, embeddings


def calculate_accuracy_at_ks(
    dataframe,
    embedding_model,
    chromadb_handler,
    max_k,
    question_field="question",
    doc_id_field="source_doc",
    batch_size=32,
    embed_document_method="embed_documents",
    query_instruction=None,
    max_length=None,
):
    """
    Calculates retrieval accuracy for Ks from 1 to max_k using ChromaDB.

    Args:
        dataframe (pd.DataFrame): The dataset.
        embedding_model: The embedding model to use.
        chromadb_handler: Instance of ChromaDBHandler.
        max_k (int): Maximum value of K for top-K retrieval.
        question_field (str): Name of the question field.
        doc_id_field (str): Name of the document ID field.
        batch_size (int): Batch size for processing questions.
        embed_document_method (str): The method name to embed queries.
        query_instruction (str): Instruction prefix for query embedding.

    Returns:
        Dict[int, float]: Dictionary mapping K to accuracy at K.
    """
    total = len(dataframe)
    ks = range(1, max_k + 1)
    correct_counts = {k: 0 for k in ks}

    # Get list of actual doc IDs
    actual_doc_ids = dataframe[doc_id_field].tolist()
    log_cuda_memory_usage("Before calculating accuracies at Ks")
    questions = dataframe[question_field].tolist()

    if hasattr(embedding_model, embed_document_method):
        embed_func = getattr(embedding_model, embed_document_method)
        if query_instruction:
            instruction_pairs = [f"{query_instruction}{text}" for text in questions]
            embeddings = embed_func(instruction_pairs)
        else:
            embeddings = embed_func(questions)
    else:
        embeddings = embedding_model.embed(
            questions,
            batch_size=batch_size,
            instruction=query_instruction,
            max_length=max_length,
        )
        embeddings = embeddings.cpu().numpy()

    torch.cuda.empty_cache()
    gc.collect()
    log_cuda_memory_usage("After computing question embeddings")

    # Query ChromaDB with the query embeddings
    results = chromadb_handler.query(
        query_embeddings=embeddings,
        n_results=max_k,
    )

    # Evaluate accuracies for each K
    for idx_in_batch, actual_doc_id in enumerate(actual_doc_ids):
        retrieved_metadatas = results["metadatas"][idx_in_batch]
        retrieved_doc_ids = [metadata["doc_id"] for metadata in retrieved_metadatas]

        unique_retrieved_doc_ids = []
        for doc_id in retrieved_doc_ids:
            if doc_id not in unique_retrieved_doc_ids:
                unique_retrieved_doc_ids.append(doc_id)
        for k in ks:
            if actual_doc_id in unique_retrieved_doc_ids[:k]:
                correct_counts[k] += 1

    del embeddings
    torch.cuda.empty_cache()
    gc.collect()
    log_cuda_memory_usage("After processing batch")

    # Calculate accuracy for each K
    accuracies = {k: correct_counts[k] / total for k in ks}
    log_cuda_memory_usage("After calculating accuracies at Ks")
    return accuracies


def run_evaluation(models, datasets, max_k):
    """
    Runs the evaluation over multiple models and datasets.

    Args:
        models (List[Dict]): List of models with settings.
        datasets (List[Dict]): List of dataset configurations.
        max_k: max K value for top-K retrieval.

    Returns:
        List[Dict]: List of results.
    """
    results = []
    for dataset_info in datasets:
        dataset_path = dataset_info["path"]
        context_field = dataset_info.get("context_field", "context")
        question_field = dataset_info.get("question_field", "question")
        doc_id_field = dataset_info.get("doc_id_field", "source_doc")
        df = load_dataset(dataset_path)
        for model_info in models:
            model_name = model_info["name"]
            get_model_func = model_info.get("get_model_func")
            batch_size = model_info.get("batch_size", 32)
            embed_document_method = model_info.get(
                "embed_document_method", "embed_documents"
            )
            instruction = model_info.get("instruction", None)
            query_instruction = model_info.get("query_instruction", None)

            logger.info(f"Evaluating model: {model_name}")
            try:
                if get_model_func:
                    embedding_model = get_model_func()
                else:
                    embedding_model = HuggingFaceEmbedding(
                        model_name=model_name,
                        load_in_8bit=model_info.get("load_in_8bit", False),
                        trust_remote_code=model_info.get("trust_remote_code", False),
                        device_map="auto",
                        max_memory={0: "18000MB", "cpu": "18000MB"},
                        **model_info.get("model_kwargs", {}),
                    )

                log_cuda_memory_usage("After loading embedding model")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                continue

            chunked_texts, document_ids, context_embeddings = (
                create_document_embeddings(
                    embedding_model,
                    df,
                    context_field,
                    doc_id_field,
                    batch_size=batch_size,
                    embed_document_method=embed_document_method,
                    instruction=instruction,
                )
            )

            # Create unique IDs for each chunk
            ids = [f"{doc_id}_{idx}" for idx, doc_id in enumerate(document_ids)]
            # Create metadatas
            metadatas = [{"doc_id": doc_id} for doc_id in document_ids]

            # Initialize ChromaDBHandler
            chromadb_handler = ChromaDBHandler(
                collection_name=alphanumeric_string(model_name)
            )

            # Add embeddings to ChromaDB
            chromadb_handler.add_embeddings(
                embeddings=context_embeddings,
                documents=chunked_texts,
                metadatas=metadatas,
                ids=ids,
            )

            # Calculate accuracies for all Ks at once
            accuracies = calculate_accuracy_at_ks(
                df,
                embedding_model,
                chromadb_handler,
                max_k,
                question_field,
                doc_id_field,
                batch_size=batch_size,
                embed_document_method=embed_document_method,
                query_instruction=query_instruction,
            )

            # Delete the collection
            chromadb_handler.delete_collection()

            result = {
                "model": model_name,
                "dataset": dataset_path,
                "settings": {
                    "instruction": instruction,
                    "query_instruction": query_instruction,
                    "load_in_8bit": model_info.get("load_in_8bit", False),
                    "trust_remote_code": model_info.get("trust_remote_code", False),
                },
            }
            for K in range(1, max_k + 1):
                accuracy = accuracies[K]
                result[f"Accuracy@k={K}"] = round(accuracy * 100, 2)
                logger.info(f"{model_name} - K={K}, Accuracy: {accuracy * 100:.2f}%")
            results.append(result)
            if hasattr(embedding_model, "clean_up"):
                embedding_model.clean_up()
            log_cuda_memory_usage("After cleaning up embedding model")
            torch.cuda.empty_cache()
            gc.collect()
    return results


if __name__ == "__main__":
    models = [
        # {
        #     "name": "nvidia/NV-Embed-v2",
        #     "batch_size": 5,
        #     "trust_remote_code": True,
        #     "load_in_8bit": True,
        #     "instruction": "Instruct: Represent this passage for retrieval in response to relevant technical questions.\nQuery:",
        #     "query_instruction": "Instruct: Given a technical query, find the most relevant passages that can provide the answer.\nPassage:",
        #     "max_length": 32768,
        # },
        # {
        #     "name": "dunzhang/stella_en_1.5B_v5",
        #     "batch_size": 25,
        #     "trust_remote_code": True,
        #     "instruction": "Instruct: Represent this passage for retrieval in response to relevant technical questions.\nQuery:",
        #     "query_instruction": "Instruct: Given a technical query, find the most relevant passages that can provide the answer.\nPassage:",
        # },
        {
            "name": "sentence-transformers/all-MiniLM-L6-v2",
            "batch_size": 100,
        },

        # {
        #     "name": "mixedbread-ai/mxbai-embed-large-v1",
        #     "batch_size": 100,
        # },
        # {
        #     "name": "MSMARCO",
        #     "get_model_func": get_msmarco_embeddings,
        #     "embed_document_method": "embed_documents",
        # },
    ]

    datasets = [
        {
            "path": "/home/ubuntu/Multi-Agent-LLM-System-with-LangGraph-RAG-and-LangChain/src/data_mining/filtered_dataset.csv",
            "context_field": "context",
            "question_field": "Question",
            "doc_id_field": "PDF Reference",
        },
    ]

    max_k = 5
    results = run_evaluation(models, datasets, max_k)
    results_df = pd.DataFrame(results)
    accuracy_columns = [f"Accuracy@k={k}" for k in range(1, max_k + 1)]
    results_df = results_df[["model", "dataset", "settings"] + accuracy_columns]
    results_df.to_excel(
        "retriever_eval_filtered_dataset_multiple_docs_per_question.xlsx",
        index=False,
    )
