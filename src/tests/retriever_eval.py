import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from src.my_rag_ollama.get_embedding_function import (
    get_msmarco_embeddings,
    get_biobert_embeddings,
    get_bert_base_uncased_embeddings,
    get_roberta_base_embeddings,
    get_instructor_xl_embeddings,
    get_roberta_large_embeddings,
    get_bert_large_nli_embeddings,
    get_mxbai_embed_large_embeddings,
)
from src.my_rag.components.embeddings.huggingface_embedding import HuggingFaceEmbedding
import logging
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def log_cuda_memory_usage(message=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        logger.info(
            f"{message} - CUDA memory allocated: {allocated:.2f} MB, reserved: {reserved:.2f} MB"
        )
    else:
        logger.info(f"{message} - CUDA not available")


def load_dataset(
    dataset_path,
):
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
    batch_size=2,
):
    """
    Creates embeddings for the documents.

    Args:
        embedding_model: The embedding model to use.
        dataframe (pd.DataFrame): The dataset.
        context_field (str): Name of the context field.
        doc_id_field (str): Name of the document ID field.
        batch_size (int): Batch size for embedding contexts.

    Returns:
        Tuple[np.ndarray, List]: Document embeddings (as numpy array) and their IDs.
    """
    contexts = dataframe[context_field].tolist()
    document_ids = dataframe[doc_id_field].tolist()
    embeddings_list = []
    log_cuda_memory_usage("Before creating context embeddings")
    embeddings = embedding_model.embed(contexts, batch_size=batch_size)
    embeddings = embeddings.cpu().numpy()
    embeddings_list.append(embeddings)
    del embeddings
    torch.cuda.empty_cache()
    gc.collect()
    context_embeddings = np.vstack(embeddings_list)
    log_cuda_memory_usage("After creating all context embeddings")
    return context_embeddings, document_ids


def calculate_accuracy_at_ks(
    dataframe,
    context_embeddings,
    document_ids,
    embedding_model,
    max_k,
    question_field="question",
    doc_id_field="source_doc",
    batch_size=2,
):
    """
    Calculates retrieval accuracy for Ks from 1 to max_k.

    Args:
        dataframe (pd.DataFrame): The dataset.
        context_embeddings (np.ndarray): Embeddings of the contexts (on CPU).
        document_ids (List): IDs of the documents.
        embedding_model: The embedding model to use.
        max_k (int): Maximum value of K for top-K retrieval.
        question_field (str): Name of the question field.
        doc_id_field (str): Name of the document ID field.
        batch_size (int): Batch size for processing questions.

    Returns:
        Dict[int, float]: Dictionary mapping K to accuracy at K.
    """
    total = len(dataframe)
    ks = range(1, max_k + 1)
    correct_counts = {k: 0 for k in ks}

    # Get list of actual doc IDs
    actual_doc_ids = dataframe[doc_id_field].tolist()
    log_cuda_memory_usage("Before calculating accuracies at Ks")
    # Compute embeddings for batch of questions
    question_embeddings = embedding_model.embed(
        dataframe[question_field].to_list(), batch_size=batch_size
    )
    question_embeddings = question_embeddings.cpu().numpy()
    torch.cuda.empty_cache()
    gc.collect()
    log_cuda_memory_usage(f"After computing question embeddings for batch")

    # Compute similarities between question embeddings and context embeddings
    similarities = cosine_similarity(question_embeddings, context_embeddings)

    # Get indices of top max_k similar contexts for each question in the batch
    top_k_indices = np.argsort(similarities, axis=1)[:, -max_k:][
        :, ::-1
    ]  # Shape: (batch_size, max_k)

    # Map indices to document IDs
    top_k_doc_ids = [
        [document_ids[idx] for idx in indices] for indices in top_k_indices
    ]

    # Evaluate accuracies for each K
    for idx, actual_doc_id in enumerate(actual_doc_ids):
        retrieved_doc_ids = top_k_doc_ids[idx]
        for k in ks:
            if actual_doc_id in retrieved_doc_ids[:k]:
                correct_counts[k] += 1

    del question_embeddings
    del similarities
    torch.cuda.empty_cache()
    gc.collect()

    accuracies = {k: correct_counts[k] / total for k in ks}
    log_cuda_memory_usage("After calculating accuracies at Ks")
    return accuracies


def run_evaluation(models, datasets, max_k, settings):
    """
    Runs the evaluation over multiple models, datasets, and configurations.

    Args:
        models (List[str]): List of model names.
        datasets (List[Dict]): List of dataset configurations.
        max_k: max K value for top-K retrieval.
        settings (List[Dict]): List of settings for model loading.

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
        for model_name in models:
            for setting in settings:
                logger.info(f"Evaluating model: {model_name}, settings: {setting}")
                try:
                    embedding_model = HuggingFaceEmbedding(
                        model_name=model_name,
                        load_in_8bit=setting.get("load_in_8bit", False),
                        trust_remote_code=setting.get("trust_remote_code", False),
                        device_map="auto",
                        max_memory={0: "18000MB", "cpu": "18000MB"},
                        **setting.get("model_kwargs", {}),
                    )
                    log_cuda_memory_usage("After loading embedding model")
                except Exception as e:
                    logger.error(
                        f"Failed to load model {model_name} with settings {setting}: {e}"
                    )
                    continue
                context_embeddings, document_ids = create_document_embeddings(
                    embedding_model, df, context_field, doc_id_field, batch_size=2
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
                    batch_size=2,  # Adjust batch size as needed
                )
                for K in range(1, max_k + 1):
                    accuracy = accuracies[K]
                    result = {
                        "model": model_name,
                        "dataset": dataset_path,
                        "K": K,
                        "accuracy": accuracy,
                        "settings": setting,
                    }
                    logger.info(f"K={K}, Accuracy: {accuracy * 100:.2f}%")
                    results.append(result)
                embedding_model.clean_up()
                log_cuda_memory_usage("After cleaning up embedding model")
                torch.cuda.empty_cache()
                gc.collect()
    return results


if __name__ == "__main__":
    models = [
        # "nvidia/NV-Embed-v2",
        "sentence-transformers/all-MiniLM-L6-v2",
        "dunzhang/stella_en_1.5B_v5",
        "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
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

    settings = [
        {"load_in_8bit": False, "trust_remote_code": False},
        {"load_in_8bit": False, "trust_remote_code": True},
        # {"load_in_8bit": True, "trust_remote_code": False},
        # {"load_in_8bit": True, "trust_remote_code": True},
    ]

    results = run_evaluation(models, datasets, max_k, settings)
    # Save results to a file
    results_df = pd.DataFrame(results)
    results_df.to_excel("retriever_eval.xlsx", index=False)
