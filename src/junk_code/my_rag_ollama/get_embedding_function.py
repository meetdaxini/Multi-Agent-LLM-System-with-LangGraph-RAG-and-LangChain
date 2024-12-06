from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings


# Function to get msmarco-distilbert-base-tas-b embeddings
def get_msmarco_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/msmarco-distilbert-base-tas-b")
    return embeddings


# Function to get biobert-v1.1 embeddings
def get_biobert_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="dmis-lab/biobert-v1.1")
    return embeddings


# Function to get bert-base-uncased embeddings
def get_bert_base_uncased_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="bert-base-uncased")
    return embeddings


# Function to get roberta-base embeddings
def get_roberta_base_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="roberta-base")
    return embeddings

# Function to get instructor-xl embeddings
def get_instructor_xl_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl")
    return embeddings


# Function to get roberta-large embeddings
def get_roberta_large_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="roberta-large")
    return embeddings


# Function to get bert-large-nli embeddings
def get_bert_large_nli_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/bert-large-nli-stsb-mean-tokens")
    return embeddings


# Function to get mxbai-embed-large embeddings
def get_mxbai_embed_large_embeddings():
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    return embeddings
