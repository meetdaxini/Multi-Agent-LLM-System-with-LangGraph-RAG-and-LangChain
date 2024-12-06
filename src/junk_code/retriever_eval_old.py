import pandas as pd
import pandas as pd
import numpy as np
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

df = pd.read_parquet(
    "hf://datasets/m-ric/huggingface_doc_qa_eval/data/train-00000-of-00001.parquet"
)


embeddings_model = get_mxbai_embed_large_embeddings()


def create_document_embeddings(dataframe):
    document_ids = []
    contexts = []
    for index, row in dataframe.iterrows():
        contexts.append(row["context"])
        document_ids.append(row["source_doc"])

    return embeddings_model.embed_documents(contexts), document_ids


context_embeddings, document_ids = create_document_embeddings(df)


def retrieve_top_document(question, context_embeddings, document_ids):
    # Generate embedding for the question
    question_embedding = embeddings_model.embed_query(
        f"Represent this sentence for searching relevant passages: {question}"
    )
    # Calculate cosine similarity between the question and all contexts
    similarities = cosine_similarity([question_embedding], context_embeddings)[0]
    # Get the index of the most similar context
    top_index = np.argmax(similarities)
    return document_ids[top_index]


def calculate_accuracy(dataframe, context_embeddings, document_ids):
    correct_count = 0
    total = len(dataframe)

    for index, row in dataframe.iterrows():
        question = row["question"]
        actual_doc_id = row["source_doc"]

        # Retrieve top matching document ID
        retrieved_doc_id = retrieve_top_document(
            question, context_embeddings, document_ids
        )

        # Check if the retrieved document ID matches the actual document ID
        if retrieved_doc_id == actual_doc_id:
            correct_count += 1
        print(question)
        print(actual_doc_id)
        print(retrieved_doc_id)
        print(correct_count)
    # Calculate accuracy
    accuracy = correct_count / total
    return accuracy


accuracy = calculate_accuracy(df, context_embeddings, document_ids)
print(f"Retrieval k=1 Accuracy: {accuracy * 100:.2f}%")


embedding_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    #         load_in_8bit=True,
    trust_remote_code=True,
    max_memory={0: "22000MB", "cpu": "22000MB"},
)


def create_document_embeddings(dataframe):
    document_ids = []
    contexts = []
    for index, row in dataframe.iterrows():
        contexts.append(row["context"])
        document_ids.append(row["source_doc"])

    return embedding_model.embed(contexts, batch_size=1, instruction=""), document_ids


context_embeddings, document_ids = create_document_embeddings(df)


def retrieve_top_document(question, context_embeddings, document_ids):
    query_prefix = "Instruct: Given a question, retrieve most relavant passages that contains answer to the question \nQuery: "
    question_embedding = embedding_model.embed([question], instruction=query_prefix)
    similarities = cosine_similarity(question_embedding, context_embeddings)[0]
    top_index = np.argmax(similarities)
    return document_ids[top_index]


def calculate_accuracy(dataframe, context_embeddings, document_ids):
    correct_count = 0
    total = len(dataframe)

    for index, row in dataframe.iterrows():
        question = row["question"]
        actual_doc_id = row["source_doc"]
        retrieved_doc_id = retrieve_top_document(
            question, context_embeddings, document_ids
        )
        if retrieved_doc_id == actual_doc_id:
            correct_count += 1

    return accuracy


# Step 6: Run the accuracy calculation
accuracy = calculate_accuracy(df, context_embeddings, document_ids)
print(f"Retrieval k=1 Accuracy hf: {accuracy * 100:.2f}%")
