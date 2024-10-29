import pandas as pd
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import logging
import gc
from my_rag_ollama.get_embedding_function import (
    get_msmarco_embeddings,
    get_mxbai_embed_large_embeddings,
)
from my_rag.components.embeddings.huggingface_embedding import HuggingFaceEmbedding
import logging
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def chunk_text(text, max_chunk_length=512, overlap=50):
    """
    Splits text into chunks of max_chunk_length with overlap.

    Args:
        text (str): The text to be chunked.
        max_chunk_length (int): The maximum length of each chunk (in words).
        overlap (int): The number of overlapping words between chunks.

    Returns:
        List[str]: List of text chunks.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_chunk_length, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap  # Move start back by overlap words
        if start < 0:
            start = 0
    return chunks


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
    chunk_size=512,
    chunk_overlap=50,
):
    """
    Creates embeddings for the documents, with chunking.

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
        Tuple[np.ndarray, List]: Document embeddings (as numpy array) and their IDs.
    """
    contexts = dataframe[context_field].tolist()
    document_ids = dataframe[doc_id_field].tolist()

    chunked_texts = []
    chunked_doc_ids = []
    for context, doc_id in zip(contexts, document_ids):
        chunks = chunk_text(context, max_chunk_length=chunk_size, overlap=chunk_overlap)
        chunked_texts.extend(chunks)
        chunked_doc_ids.extend([doc_id] * len(chunks))

    print(len(chunked_texts))
    print(len(chunked_doc_ids))
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

    return embeddings, chunked_doc_ids


def calculate_accuracy_at_ks(
    dataframe,
    context_embeddings,
    chunked_doc_ids,
    embedding_model,
    max_k,
    question_field="question",
    doc_id_field="source_doc",
    batch_size=32,
    embed_document_method="embed_documents",
    query_instruction=None,
    max_length=None,
):
    """
    Calculates retrieval accuracy for Ks from 1 to max_k.

    Args:
        dataframe (pd.DataFrame): The dataset.
        context_embeddings (np.ndarray): Embeddings of the contexts (on CPU).
        chunked_doc_ids (List): Document IDs corresponding to each chunk.
        embedding_model: The embedding model to use.
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

    # Compute similarities between question embeddings and context embeddings
    similarities = cosine_similarity(embeddings, context_embeddings)

    # Get indices of top max_k similar contexts for each question
    top_k_indices = np.argsort(similarities, axis=1)[:, -max_k:]  # (num_queries, max_k)

    # Map indices to document IDs
    top_k_doc_ids_per_query = []
    for indices in top_k_indices:
        retrieved_doc_ids = [chunked_doc_ids[idx] for idx in indices]
        top_k_doc_ids_per_query.append(retrieved_doc_ids)

    # Evaluate accuracies for each K
    for idx_in_batch, actual_doc_id in enumerate(actual_doc_ids):
        retrieved_doc_ids = top_k_doc_ids_per_query[idx_in_batch]
        unique_retrieved_doc_ids = []
        for doc_id in retrieved_doc_ids:
            if doc_id not in unique_retrieved_doc_ids:
                unique_retrieved_doc_ids.append(doc_id)
        for k in ks:
            if actual_doc_id in unique_retrieved_doc_ids[:k]:
                correct_counts[k] += 1
                break  # Correct for this K and all higher Ks

    del embeddings
    del similarities
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
            context_embeddings, document_ids = create_document_embeddings(
                embedding_model,
                df,
                context_field,
                doc_id_field,
                batch_size=batch_size,
                embed_document_method=embed_document_method,
                instruction=instruction,
            )

            # Calculate accuracies for all Ks at once
            accuracies = calculate_accuracy_at_ks(
                df,
                context_embeddings,
                document_ids,
                embedding_model,
                max_k,
                question_field,
                doc_id_field,
                batch_size=batch_size,
                embed_document_method=embed_document_method,
                query_instruction=query_instruction,
            )
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
        #     "batch_size": 1,
        #     "trust_remote_code": True,
        #     "load_in_8bit": True,
        #     "instruction": "Instruct: Represent this passage for retrieval in response to relevant technical questions.\nQuery:",
        #     "query_instruction": "Instruct: Given a technical query, find the most relevant passages that can provide the answer.\nPassage:",
        #     "max_length": 32768,
        # },
        # {
        #     "name": "dunzhang/stella_en_1.5B_v5",
        #     "batch_size": 2,
        #     "trust_remote_code": True,
        #     "instruction": "Instruct: Represent this passage for retrieval in response to relevant technical questions.\nQuery:",
        #     "query_instruction": "Instruct: Given a technical query, find the most relevant passages that can provide the answer.\nPassage:",
        # },
        {
            "name": "sentence-transformers/all-MiniLM-L6-v2",
            "batch_size": 2,
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

    max_k = 5
    results = run_evaluation(models, datasets, max_k)
    results_df = pd.DataFrame(results)
    accuracy_columns = [f"Accuracy@k={k}" for k in range(1, max_k + 1)]
    results_df = results_df[["model", "dataset", "settings"] + accuracy_columns]
    results_df.to_excel("retriever_eval.xlsx", index=False)
