import pandas as pd
import torch
import logging
import gc
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from my_rag_ollama.get_embedding_function import get_mxbai_embed_large_embeddings
from my_rag.components.embeddings.huggingface_embedding import HuggingFaceEmbedding
from my_rag.components.llms.huggingface_llm import HuggingFaceLLM  # Add this line
from ragatouille import RAGPretrainedModel
import chromadb
import re
import os
import math

DEFAULT_DATA_PATH = "data_test"  # Path to your PDFs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

rag_model = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")


class Question:
    def __init__(self, text, rag_model, llm, k=5):
        self.text = text
        self.rag_model = rag_model
        self.llm = llm
        self.k = k
        self.retrieved_documents = []
        self.answer = None

    def retrieve_documents(self):
        """Retrieve relevant documents using the rag_model."""
        search_results = self.rag_model.search(self.text, k=self.k)
        # Extract text content from each document in search results
        self.retrieved_documents = [doc.get("text", "") if isinstance(doc, dict) else doc for doc in search_results]
        return self.retrieved_documents


    def generate_answer(self):
        """Generate an answer using the LLM with retrieved documents as context."""
        if not self.retrieved_documents:
            raise ValueError("No documents retrieved. Call 'retrieve_documents' first.")

        # Assuming retrieved_documents is a list of document texts
        context = "\n\n".join(self.retrieved_documents)

        # Generate the answer using the context and question text
        self.answer = self.llm.generate_response_with_context(
            context=context,
            prompt=self.text,
            temperature=0.7,
            top_p=0.9,
            num_beams=3,
            early_stopping=True
        )
        return self.answer


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

    def verify_population(self):
        num_documents = self.collection.count()
        logger.info(f"Collection '{self.collection_name}' has {num_documents} documents.")
        if num_documents == 0:
            raise RuntimeError("ChromaDB collection is empty. Embeddings were not added successfully.")


def process_questions_in_batches(questions, rag_model, llm, k=5, batch_size=8):
    results = []
    num_batches = math.ceil(len(questions) / batch_size)

    for i in range(num_batches):
        batch_questions = questions[i * batch_size:(i + 1) * batch_size]
        logger.info(f"Processing batch {i + 1}/{num_batches} with {len(batch_questions)} questions.")

        for question_text in batch_questions:
            question = Question(text=question_text, rag_model=rag_model, llm=llm, k=k)
            question.retrieve_documents()
            answer = question.generate_answer()

            results.append({
                "question": question_text,
                "retrieved_documents": question.retrieved_documents,
                "answer": answer
            })

        # Clear memory between batches if necessary
        torch.cuda.empty_cache()
        gc.collect()

    return results



def alphanumeric_string(input_string):
    return re.sub(r"[^a-zA-Z0-9]", "", input_string)


def log_cuda_memory_usage(message=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        logger.info(f"{message} - CUDA memory allocated: {allocated:.2f} MB, reserved: {reserved:.2f} MB")
    else:
        logger.info(f"{message} - CUDA not available")


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


def create_document_embeddings_with_colbert(
        rag_model,
        dataframe,
        context_field="context",
        doc_id_field="source_doc",
        batch_size=32,
        chunk_size=1000,
        chunk_overlap=115
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

    # Encode documents using ColBERT
    embeddings = rag_model.index_documents(chunked_texts)

    torch.cuda.empty_cache()
    gc.collect()
    log_cuda_memory_usage("After processing context embeddings")

    return chunked_texts, chunked_doc_ids, embeddings


def generate_answers_with_colbert(questions, rag_model, llm, k=5):
    answers, retrieved_contexts_list = [], []

    for question_text in questions:
        # Retrieve documents using ColBERT
        retrieved_documents = rag_model.search(question_text, k=k)
        # Assuming retrieved_documents is a list of texts
        retrieved_contexts = "\n\n".join(retrieved_documents)

        # Generate the answer using the LLM
        answer = llm.generate_response_with_context(
            context=retrieved_contexts, prompt=question_text, max_new_tokens=100, temperature=0.7, top_p=0.9
        )

        answers.append(answer)
        retrieved_contexts_list.append(retrieved_contexts)

    results_df = pd.DataFrame({
        "question": questions,
        "Retrieved_Contexts": retrieved_contexts_list,
        "LLM_Answer": answers
    })

    return results_df


def run_evaluation(rag_model, datasets, k):
    # Initialize your LLM
    llm = HuggingFaceLLM(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float16,
        load_in_8bit=True,
        device_map="auto",
        trust_remote_code=True,
    )

    for dataset_info in datasets:
        # Load the dataset
        df = load_dataset(dataset_info["path"])

        # Extract texts from the DataFrame
        my_documents = df['context'].tolist()

        # Index the documents using ColBERT
        index_name = "my_index"
        rag_model.index(index_name=index_name, collection=my_documents)

        # Define your questions from the dataset
        questions = df['question'].tolist()

        # Generate answers using ColBERT
        df_with_answers = generate_answers_with_colbert(
            questions=questions,
            rag_model=rag_model,
            llm=llm,
            k=k
        )

        # Optionally, you can compare retrieved documents with actual documents if your dataset has that information
        if 'source_doc' in df.columns:
            df_with_answers['Actual_Doc_ID'] = df['source_doc'].tolist()

        # Save the results
        output_filename = f"retriever_eval_{dataset_info['name']}.xlsx"
        df_with_answers.to_excel(output_filename, index=False)
        logger.info(f"Results saved to {output_filename}")

    llm.clean_up()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load your dataset
    df = load_dataset()  # Ensure your PDFs are in the 'data_test' directory

    # Extract texts from the DataFrame
    my_documents = df['context'].tolist()

    # Initialize the ColBERT-based RAG model
    rag_model = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

    # Create an index with your documents
    index_name = "my_index"
    rag_model.index(index_name=index_name, collection=my_documents)

    # Initialize your LLM
    llm = HuggingFaceLLM(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float16,
        load_in_8bit=True,
        device_map="auto",
        trust_remote_code=True,
    )

    # Define your questions
    questions = [
        "Is Hirschsprung disease a mendelian or a multifactorial disorder?",
        "List signaling molecules (ligands) that interact with the receptor EGFR?",
        "Are long non coding RNAs spliced?",
        "Is RANKL secreted from the cells?",
        "Which miRNAs could be used as potential biomarkers for epithelial ovarian cancer?",
        "Which are the Yamanaka factors?",
        "Is the monoclonal antibody Trastuzumab (Herceptin) of potential use in the treatment of prostate cancer?"
    ]

    # Process questions in batches
    results = process_questions_in_batches(
        questions=questions,
        rag_model=rag_model,
        llm=llm,
        k=5,
        batch_size=8
    )

    # Print results
    for i, result in enumerate(results, start=1):
        print(f"{i}. Question: {result['question']}")
        print(f"Answer: {result['answer']}\n")





