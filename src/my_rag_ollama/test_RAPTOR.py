import pandas as pd
import pandas as np
import torch
import logging
import gc
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from my_rag.components.embeddings.huggingface_embedding import HuggingFaceEmbedding
from my_rag.components.llms.huggingface_llm import HuggingFaceLLM
from concurrent.futures import ThreadPoolExecutor, as_completed
import chromadb
import re
import os
import math
import umap.umap_ as umap  # Import UMAP
from tqdm import tqdm  # Import tqdm for progress tracking

DEFAULT_DATA_PATH = "data_test"  # Path to your PDFs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAPTORPipeline:
    def __init__(self, embedding_model, llm, chromadb_handler, depth=3, n_clusters=5):
        """
        Parameters:
            embedding_model (object): Embedding model.
            llm (object): Language model for generating answers.
            chromadb_handler (ChromaDBHandler): Vector DB handler.
            depth (int): Number of recursive abstraction levels.
            n_clusters (int): Initial number of clusters per level.
        """
        self.embedding_model = embedding_model
        self.llm = llm
        self.chromadb_handler = chromadb_handler
        self.depth = depth
        self.n_clusters = n_clusters
        self.hierarchy = {}

    def evaluate_clusters(self, embeddings, max_clusters=10):
        """Evaluate clustering to determine the optimal number of clusters using silhouette score."""
        silhouette_scores = []
        cluster_range = range(2, max_clusters + 1)
        for n_clusters in cluster_range:
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = clustering.fit_predict(embeddings)
            if len(set(cluster_labels)) > 1:
                score = silhouette_score(embeddings, cluster_labels)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(-1)
        best_n_clusters = cluster_range[silhouette_scores.index(max(silhouette_scores))]
        logger.info(f"Optimal number of clusters: {best_n_clusters} with silhouette score: {max(silhouette_scores)}")
        return best_n_clusters

    def recursive_abstraction(self, texts, level=0):
        """Recursively clusters and summarizes documents into hierarchical abstractions."""
        if level >= self.depth or len(texts) <= 1:
            return texts

        # Embed the texts
        embeddings = self.embedding_model.embed(texts)
        embeddings = embeddings.cpu().numpy()

        # Reduce dimensionality with UMAP
        reducer = umap.UMAP(n_components=50)
        reduced_embeddings = reducer.fit_transform(embeddings)
        logger.info(f"UMAP reduced embeddings to shape: {reduced_embeddings.shape}")

        # Determine the optimal number of clusters
        best_n_clusters = self.evaluate_clusters(reduced_embeddings, max_clusters=min(10, len(texts) - 1))

        # Perform clustering with the optimal number of clusters
        clustering = AgglomerativeClustering(n_clusters=best_n_clusters).fit(reduced_embeddings)
        cluster_labels = clustering.labels_

        # Aggregate texts by cluster labels to create summaries
        clustered_texts = {i: [] for i in range(best_n_clusters)}
        for label, text in zip(cluster_labels, texts):
            clustered_texts[label].append(text)

        summaries = []
        for cluster_id, cluster_texts in clustered_texts.items():
            # Join the texts
            context = "\n\n".join(cluster_texts)

            # Tokenize and truncate to fit within max_length - space for response
            max_context_tokens = 200  # Adjust as needed, ensuring enough space for response
            tokens = self.llm.tokenizer(context, return_tensors="pt")["input_ids"]
            if tokens.shape[1] > max_context_tokens:
                truncated_context = self.llm.tokenizer.decode(tokens[0, -max_context_tokens:], skip_special_tokens=True)
            else:
                truncated_context = context

            # Generate a summary for each cluster
            summary = self.llm.generate_response_with_context(
                context=truncated_context,
                prompt=f"Summarize level {level + 1} cluster {cluster_id}",
                max_length=256,
                temperature=0.5,
                top_p=0.9
            )
            summaries.append(summary)

        return summaries

    def add_hierarchy_to_db(self, texts, doc_ids):
        """Add the recursive abstraction hierarchy to the vector DB."""
        final_summaries = self.recursive_abstraction(texts)

        if not final_summaries:
            logger.error("No summaries generated for the hierarchy.")
            return

        # Create new doc_ids list matching the length of summaries
        adjusted_doc_ids = doc_ids[:len(final_summaries)]
        if len(final_summaries) != len(doc_ids):
            logger.warning(f"Adjusted document IDs from {len(doc_ids)} to {len(final_summaries)} to match summaries")

        # Generate embeddings for the summaries
        embeddings = self.embedding_model.embed(final_summaries)
        embeddings = embeddings.cpu().numpy()

        # Prepare metadatas and ids for each embedding
        metadatas = [{"doc_id": doc_id} for doc_id in adjusted_doc_ids]
        ids = [f"{doc_id}_{level}" for level, doc_id in enumerate(adjusted_doc_ids)]

        self.chromadb_handler.add_embeddings(
            embeddings=embeddings,
            documents=final_summaries,
            metadatas=metadatas,
            ids=ids
        )

    def retrieve_documents(self, query):
        """Query vector DB by navigating through the abstract hierarchy."""
        query_embedding = self.embedding_model.embed([query])[0].cpu().numpy()
        results = self.chromadb_handler.query(query_embeddings=[query_embedding], n_results=5)

        return results["documents"][0] if "documents" in results else []


class Question:
    def __init__(self, text, embedding_model, chromadb_handler, llm, k=5):
        """
        Initialize a Question object.

        Parameters:
            text (str): The question text.
            embedding_model (object): The model for generating question embeddings.
            chromadb_handler (ChromaDBHandler): Handler for querying the vector DB.
            llm (object): The language model for generating answers.
            k (int): Number of relevant documents to retrieve.
        """
        self.text = text
        self.embedding_model = embedding_model
        self.chromadb_handler = chromadb_handler
        self.llm = llm
        self.k = k
        self.embedding = None
        self.retrieved_documents = []
        self.answer = None

    def embed_question(self):
        """Generate embeddings for the question text."""
        if hasattr(self.embedding_model, "embed"):
            embed_func = getattr(self.embedding_model, "embed")
            self.embedding = embed_func([self.text])[0].cpu().numpy()
        else:
            raise ValueError("Embedding model does not support the 'embed' method.")

    def retrieve_documents(self):
        """Query the vector DB to retrieve relevant documents."""
        if self.embedding is None:
            raise ValueError("Question embedding not generated. Call 'embed_question' first.")

        logger.info(f"Querying with embedding shape: {self.embedding.shape}")

        results = self.chromadb_handler.query(query_embeddings=[self.embedding], n_results=self.k)

        if "documents" not in results or not results["documents"][0]:
            raise ValueError(
                "No documents retrieved. Make sure the vector database is populated and query embeddings are correct.")

        self.retrieved_documents = results["documents"][0]
        logger.info(f"Retrieved {len(self.retrieved_documents)} documents for the query.")

        return self.retrieved_documents

    def generate_answer(self):
        """Generate an answer using the LLM with retrieved documents as context."""
        if not self.retrieved_documents:
            raise ValueError("No documents retrieved. Call 'retrieve_documents' first.")

        # Concatenate retrieved documents
        context = "\n\n".join(self.retrieved_documents)

        # Calculate tokens in context
        tokens = self.llm.tokenizer(context, return_tensors="pt")["input_ids"]
        context_length = tokens.shape[1]

        # Set appropriate max_length and max_new_tokens
        max_new_tokens = 512  # for the response
        max_input_tokens = 2048  # adjust based on your model's context window

        if context_length > max_input_tokens:
            # Truncate context if too long
            truncated_tokens = tokens[0, :max_input_tokens]
            context = self.llm.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            logger.warning(f"Context truncated from {context_length} to {max_input_tokens} tokens")

        # Generate the answer using the LLM
        self.answer = self.llm.generate_response_with_context(
            context=context,
            prompt=self.text,
            max_new_tokens=max_new_tokens,  # Use max_new_tokens instead of max_length
            temperature=0.7,
            top_p=0.9
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
            embeddings=embeddings.tolist(),  # Convert to list for compatibility
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
        num_documents = len(self.collection.list_documents())
        logger.info(f"Collection '{self.collection_name}' has {num_documents} documents.")
        if num_documents == 0:
            raise RuntimeError("ChromaDB collection is empty. Embeddings were not added successfully.")


def retrieve_documents_in_batches(batch_embeddings, chromadb_handler, k):
    """Retrieve documents for a batch of questions in parallel."""
    results = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(chromadb_handler.query, [embedding], k) for embedding in batch_embeddings]
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                logger.error(f"Error in document retrieval: {e}")
    return results


def process_questions_in_batches(questions, embedding_model, chromadb_handler, llm, k=5, batch_size=8):
    results = []
    num_batches = math.ceil(len(questions) / batch_size)

    for i in tqdm(range(num_batches), desc='Processing Batches'):
        batch_questions = questions[i * batch_size:(i + 1) * batch_size]
        batch_embeddings = embedding_model.embed(batch_questions).cpu().numpy()

        # Retrieve documents in parallel for batch
        batch_results = retrieve_documents_in_batches(batch_embeddings, chromadb_handler, k)

        for idx, (embedding, question_text) in enumerate(zip(batch_embeddings, batch_questions)):
            question = Question(text=question_text, embedding_model=embedding_model,
                                chromadb_handler=chromadb_handler, llm=llm, k=k)
            question.embedding = embedding
            question.retrieved_documents = batch_results[idx].get("documents", [])[0]
            question.answer = question.generate_answer()

            results.append({
                "question": question_text,
                "retrieved_documents": question.retrieved_documents,
                "answer": question.answer
            })

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


def create_document_embeddings(
        embedding_model,
        dataframe,
        context_field="context",
        doc_id_field="source_doc",
        batch_size=32,
        chunk_size=1000,
        chunk_overlap=115,
        instruction="",
        max_length=None,
):
    """
    Create embeddings for the document contexts in the dataframe.

    Parameters:
        embedding_model (object): The embedding model to use.
        dataframe (pd.DataFrame): DataFrame containing the documents.
        context_field (str): The field name for document contexts.
        doc_id_field (str): The field name for document IDs.
        batch_size (int): Batch size for embedding creation.
        chunk_size (int): Size of text chunks.
        chunk_overlap (int): Overlap between text chunks.
        instruction (str): Instruction to prepend to each text (if any).
        max_length (int): Max length for embeddings (if any).

    Returns:
        tuple: (chunked_texts, chunked_doc_ids, embeddings)
    """
    contexts = dataframe[context_field].tolist()
    document_ids = dataframe[doc_id_field].tolist()

    chunked_texts, chunked_doc_ids = [], []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Split texts into chunks
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

    # Create embeddings in batches
    embeddings_list = []
    for i in tqdm(range(0, len(chunked_texts), batch_size), desc='Embedding Documents'):
        batch_texts = chunked_texts[i:i + batch_size]
        if instruction:
            batch_texts = [f"{instruction}{text}" for text in batch_texts]
        batch_embeddings = embedding_model.embed(batch_texts)
        embeddings_list.append(batch_embeddings.cpu())

    # Concatenate embeddings into a single tensor
    embeddings = torch.cat(embeddings_list, dim=0)

    # Convert embeddings to numpy array if needed
    embeddings = embeddings.numpy()

    torch.cuda.empty_cache()
    gc.collect()
    log_cuda_memory_usage("After processing context embeddings")

    return chunked_texts, chunked_doc_ids, embeddings


def generate_answers_in_parallel(dataframe, llm, context_field="Retrieved_Contexts", question_field="question"):
    answers = []

    def generate_answer_for_row(row):
        context = row[context_field]
        question = row[question_field]
        return llm.generate_response_with_context(
            context=context, prompt=question, max_length=512, temperature=0.7, top_p=0.9
        )

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(generate_answer_for_row, row) for _, row in dataframe.iterrows()]
        for future in as_completed(futures):
            try:
                answers.append(future.result())
            except Exception as e:
                logger.error(f"Error in answer generation: {e}")

    dataframe["LLM_Answer"] = answers
    return dataframe


def run_evaluation(models, datasets, k):
    llm = HuggingFaceLLM(
        model_name="meta-llama/Meta-Llama-2-7b-chat-hf",
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float16,
        load_in_8bit=True,
        device_map="auto",
        trust_remote_code=True,
    )

    for dataset_info in datasets:
        df = load_dataset(dataset_info["path"])
        for model_info in models:
            model_name = model_info["name"]
            embedding_model = HuggingFaceEmbedding(
                model_name=model_name, load_in_8bit=model_info.get("load_in_8bit", False), device_map="auto"
            )

            chunked_texts, document_ids, context_embeddings = create_document_embeddings(
                embedding_model, df, chunk_size=1000, chunk_overlap=115
            )

            ids = [f"{doc_id}_{idx}" for idx, doc_id in enumerate(document_ids)]
            metadatas = [{"doc_id": doc_id} for doc_id in document_ids]

            chromadb_handler = ChromaDBHandler(collection_name=alphanumeric_string(model_name))
            chromadb_handler.add_embeddings(
                embeddings=context_embeddings, documents=chunked_texts, metadatas=metadatas, ids=ids
            )

            # Use the generate_answers_in_parallel function
            df_with_answers = generate_answers_in_parallel(
                dataframe=df, llm=llm, context_field="context"
            )

            chromadb_handler.delete_collection()
            output_filename = f"retriever_eval_{model_name.replace('/', '_')}.xlsx"
            df_with_answers.to_excel(output_filename, index=False)
            logger.info(f"Results saved to {output_filename}")

    llm.clean_up()


if __name__ == "__main__":
    # Load your dataset
    df = load_dataset()

    # Initialize your embedding model
    embedding_model = HuggingFaceEmbedding(model_name="dunzhang/stella_en_1.5B_v5")

    # Create document embeddings with UMAP and cluster evaluation
    chunked_texts, document_ids, context_embeddings = create_document_embeddings(
        embedding_model=embedding_model,
        dataframe=df,
        context_field="context",
        doc_id_field="source_doc",
        batch_size=32,
        chunk_size=1000,
        chunk_overlap=115
    )

    # Generate IDs and metadata for the embeddings
    ids = [f"{doc_id}_{idx}" for idx, doc_id in enumerate(document_ids)]
    metadatas = [{"doc_id": doc_id} for doc_id in document_ids]

    # Initialize ChromaDB handler
    chromadb_handler = ChromaDBHandler(collection_name="my_collection")

    # Add embeddings to ChromaDB
    chromadb_handler.add_embeddings(
        embeddings=context_embeddings,
        documents=chunked_texts,
        metadatas=metadatas,
        ids=ids
    )

    # Initialize your LLM
    llm = HuggingFaceLLM(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float16,
        load_in_8bit=True,
        device_map="auto",
        trust_remote_code=True,
    )

    # Initialize RAPTOR pipeline with the updated methods
    raptor_pipeline = RAPTORPipeline(
        embedding_model=embedding_model,
        llm=llm,
        chromadb_handler=chromadb_handler,
        depth=3  # You can adjust the depth as needed
    )

    # Add hierarchy to the database
    raptor_pipeline.add_hierarchy_to_db(chunked_texts, document_ids)

    # Define your questions
    questions = [
        "Is Hirschsprung disease a Mendelian or a multifactorial disorder?",
        "List signaling molecules (ligands) that interact with the receptor EGFR?",
        "Are long non-coding RNAs spliced?",
        "Is RANKL secreted from the cells?",
        "Which miRNAs could be used as potential biomarkers for epithelial ovarian cancer?",
        "Which are the Yamanaka factors?",
        "Is the monoclonal antibody Trastuzumab (Herceptin) of potential use in the treatment of prostate cancer?"
    ]

    # Generate answers for the list of questions
    results = process_questions_in_batches(
        questions,
        embedding_model,
        chromadb_handler,
        llm,
        k=5,
        batch_size=8
    )

    # Print the results
    for i, result in enumerate(results, start=1):
        print(f"{i}. Question: {result['question']}")
        print(f"Answer: {result['answer']}\n")