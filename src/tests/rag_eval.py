import pandas as pd
import numpy as np
import torch
import logging
import gc
from langchain.text_splitter import RecursiveCharacterTextSplitter
from my_rag_ollama.get_embedding_function import (
    get_msmarco_embeddings,
    get_mxbai_embed_large_embeddings,
)
from my_rag.components.embeddings.huggingface_embedding import HuggingFaceEmbedding
from my_rag.components.llms.huggingface_llm import HuggingFaceLLM
import chromadb
import re

from typing import Optional, Dict


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
        self.client.delete_collection(name=collection_name)
        self.collection = self.client.create_collection(name=collection_name)

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

    def query(self, query_embeddings, n_results, include=["documents", "metadatas"]):
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


def load_dataset(dataset_path):
    """
    Loads the dataset from the given path.

    Args:
        dataset_path (str): Path to the dataset file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    df = pd.read_parquet(dataset_path)
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
    chunk_size=1000,
    chunk_overlap=115,
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


def generate_answers(
    dataframe,
    embedding_model,
    chromadb_handler,
    llm,
    k,
    context_field="context",
    question_field="question",
    doc_id_field="source_doc",
    batch_size=32,
    embed_document_method="embed_documents",
    query_instruction=None,
    max_length=None,
):
    """
    Generates answers for each question using the retrieved contexts.

    Args:
        dataframe (pd.DataFrame): The dataset.
        embedding_model: The embedding model to use.
        chromadb_handler: Instance of ChromaDBHandler.
        llm: Instance of HuggingFaceLLM.
        k (int): Number of contexts to retrieve.
        context_field (str): Name of the context field.
        question_field (str): Name of the question field.
        doc_id_field (str): Name of the document ID field.
        batch_size (int): Batch size for processing questions.
        embed_document_method (str): The method name to embed queries.
        query_instruction (str): Instruction prefix for query embedding.

    Returns:
        pd.DataFrame: DataFrame with answers and retrieved contexts.
    """
    questions = dataframe[question_field].tolist()
    actual_doc_ids = dataframe[doc_id_field].tolist()
    answers = []
    retrieved_contexts_list = []
    retrieved_doc_ids_list = []

    # Embed the questions
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

    # Query ChromaDB with the question embeddings
    results = chromadb_handler.query(
        query_embeddings=embeddings.tolist(),
        n_results=k,
    )

    # For each question, generate an answer using the retrieved contexts
    for idx in range(len(questions)):
        question = questions[idx]
        retrieved_documents = results["documents"][idx]
        retrieved_metadatas = results["metadatas"][idx]
        retrieved_doc_ids = [meta["doc_id"] for meta in retrieved_metadatas]
        retrieved_contexts = "\n\n".join(retrieved_documents)

        # Generate answer using the LLM
        answer = llm.generate_response_with_context(
            context=retrieved_contexts,
            prompt=question,
            max_length=512,  # You can adjust this value
            temperature=0.7,  # You can adjust this value
            top_p=0.9,  # You can adjust this value
        )

        answers.append(answer)
        retrieved_contexts_list.append(retrieved_contexts)
        retrieved_doc_ids_list.append(retrieved_doc_ids)

    # Add the answers and retrieved contexts to the dataframe
    dataframe["Retrieved_Doc_IDs"] = retrieved_doc_ids_list
    dataframe["Retrieved_Contexts"] = retrieved_contexts_list
    dataframe["LLM_Answer"] = answers

    return dataframe


def run_evaluation(models, datasets, k):
    """
    Runs the evaluation over multiple models and datasets.

    Args:
        models (List[Dict]): List of models with settings.
        datasets (List[Dict]): List of dataset configurations.
        k (int): Number of contexts to retrieve.

    Returns:
        None
    """
    # Initialize the LLM
    llm = HuggingFaceLLM(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float16,
        load_in_8bit=True,
        device_map="auto",
        trust_remote_code=True,
    )

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
            max_length = model_info.get("max_length", None)

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

            (
                chunked_texts,
                document_ids,
                context_embeddings,
            ) = create_document_embeddings(
                embedding_model,
                df,
                context_field,
                doc_id_field,
                batch_size=batch_size,
                embed_document_method=embed_document_method,
                instruction=instruction,
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

            # Generate answers and update the dataframe
            df_with_answers = generate_answers(
                df,
                embedding_model,
                chromadb_handler,
                llm,
                k,
                context_field=context_field,
                question_field=question_field,
                doc_id_field=doc_id_field,
                batch_size=batch_size,
                embed_document_method=embed_document_method,
                query_instruction=query_instruction,
                max_length=max_length,
            )

            # Delete the collection
            chromadb_handler.delete_collection()

            # Save the dataframe to Excel
            output_filename = f"retriever_eval_{model_name.replace('/', '_')}.xlsx"
            df_with_answers.to_excel(output_filename, index=False)
            logger.info(f"Results saved to {output_filename}")

            if hasattr(embedding_model, "clean_up"):
                embedding_model.clean_up()
            log_cuda_memory_usage("After cleaning up embedding model")
            torch.cuda.empty_cache()
            gc.collect()

    # Clean up LLM
    llm.clean_up()


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
        #     "batch_size": 40,
        #     "trust_remote_code": True,
        #     "instruction": "Instruct: Represent this passage for retrieval in response to relevant technical questions.\nQuery:",
        #     "query_instruction": "Instruct: Given a technical query, find the most relevant passages that can provide the answer.\nPassage:",
        # },
        {
            "name": "sentence-transformers/all-MiniLM-L6-v2",
            "batch_size": 100,
        },
        # {
        #     "name": "MSMARCO",
        #     "get_model_func": get_msmarco_embeddings,
        #     "embed_document_method": "embed_documents",
        # },
        # {
        #     "name": "mxbai embed large",
        #     "get_model_func": get_mxbai_embed_large_embeddings,
        #     "embed_document_method": "embed_documents",
        #     "instruction": "Represent this passage for retrieval in response to relevant technical questions.\nQuery:",
        # },
    ]

    datasets = [
        {
            "path": "hf://datasets/m-ric/huggingface_doc_qa_eval/data/train-00000-of-00001.parquet",
            "context_field": "context",
            "question_field": "question",
            "doc_id_field": "source_doc",
        },
    ]

    k = 5
    run_evaluation(models, datasets, k)
