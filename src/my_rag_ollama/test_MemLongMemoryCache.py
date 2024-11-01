import pandas as pd
import torch
import logging
import gc
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from my_rag_ollama.get_embedding_function import get_mxbai_embed_large_embeddings
from my_rag.components.embeddings.huggingface_embedding import HuggingFaceEmbedding
from my_rag.components.llms.huggingface_llm import HuggingFaceLLM
from my_rag.components.memory_db.memlong_db import MemoryDB
import chromadb
import re
import os
import math

DEFAULT_DATA_PATH = "data_test"  # Path to your PDFs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        # Try to use 'embed_documents' if it exists; otherwise, use 'embed' or a custom method.
        if hasattr(self.embedding_model, "embed_documents"):
            embed_func = getattr(self.embedding_model, "embed_documents")
            self.embedding = embed_func([self.text])[0].cpu().numpy()
        elif hasattr(self.embedding_model, "embed"):
            embed_func = getattr(self.embedding_model, "embed")
            self.embedding = embed_func([self.text])[0].cpu().numpy()
        elif hasattr(self.embedding_model, "get_embeddings"):  # Custom method if defined
            embed_func = getattr(self.embedding_model, "get_embeddings")
            self.embedding = embed_func([self.text])[0].cpu().numpy()
        else:
            raise ValueError("Embedding model does not support any recognized embedding method.")

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

        # Combine context and prompt
        combined_input = f"{context}\n\nQuestion: {self.text}\n\nAnswer:"

        # Tokenize the combined input
        input_ids = self.llm.tokenizer.encode(combined_input, return_tensors="pt").to(self.llm.device)
        input_length = input_ids.size(1)

        # Check if input length exceeds model's max input length
        model_max_length = self.llm.model.config.max_position_embeddings  # e.g., 2048
        if input_length > model_max_length:
            # Truncate the context to fit within model's max input length
            logger.warning(
                f"Input length ({input_length}) exceeds model's max input length ({model_max_length}). Truncating context.")
            # Calculate how many tokens to keep from the context
            tokens_to_keep = model_max_length - (
                        len(self.llm.tokenizer.encode(f"Question: {self.text}\n\nAnswer:")) + 1)
            truncated_context_ids = input_ids[0, :tokens_to_keep]
            truncated_input_ids = torch.cat([
                truncated_context_ids,
                self.llm.tokenizer.encode(f"Question: {self.text}\n\nAnswer:", return_tensors="pt").to(self.llm.device)[
                    0]
            ])
        else:
            truncated_input_ids = input_ids[0]

        # Check if the tokenizer has a pad token ID; if not, set it to eos_token_id
        if self.llm.tokenizer.pad_token_id is None:
            self.llm.tokenizer.pad_token_id = self.llm.tokenizer.eos_token_id

        # Generate the attention mask based on pad token ID
        attention_mask = (truncated_input_ids != self.llm.tokenizer.pad_token_id).unsqueeze(0)

        # Generate the answer with the updated attention mask and pad token ID
        output = self.llm.model.generate(
            input_ids=truncated_input_ids.unsqueeze(0),
            attention_mask=attention_mask,
            max_new_tokens=512,  # Specify the number of tokens to generate
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.llm.tokenizer.pad_token_id  # Now this is explicitly set
        )

        # Decode the generated tokens
        generated_text = self.llm.tokenizer.decode(output[0], skip_special_tokens=True)

        # Extract the answer part
        self.answer = generated_text[len(combined_input):].strip()
        return self.answer


class MemoryAugmentedRAG:
    def __init__(self, embedding_model_path, embedding_dim, llm, max_memory_size=10000, k=5):
        # Load embedding model for chunk embeddings
        self.embedder = HuggingFaceEmbedding(embedding_model_path)
        self.memory_db = MemoryDB(embedding_dim=embedding_dim, max_memory_size=max_memory_size)
        self.k = k
        self.llm = llm  # Pass the language model instance to the constructor

    def embed_question(self, text):
        """Embed the input question for retrieval."""
        return self.embedder.encode([text])

    def retrieve_documents(self, query_embedding):
        """Retrieve top-k relevant documents from memory using cosine similarity."""
        return self.memory_db.retrieve_top_k(query_embedding, k=self.k)

    def prepopulate_memory(self, questions_answers):
        """
        Prepopulate the memory with question-answer pairs.

        Parameters:
            questions_answers (list of tuples): List of (question, answer) pairs.
        """
        for question, answer in questions_answers:
            # Generate embedding for the answer
            answer_embedding = self.embedder.encode([answer])[0]

            # Add to memory DB
            self.memory_db.add_to_memory(answer_embedding, answer)

    def generate_answer(self, question_text):
        """Generate an answer using the retrieved documents as context."""
        # Embed the query
        query_embedding = self.embed_question(question_text)

        # Retrieve relevant documents
        retrieved_docs = self.retrieve_documents(query_embedding)

        # Combine the documents into a single context
        context = "\n\n".join(retrieved_docs)
        combined_input = f"{context}\n\nQuestion: {question_text}\n\nAnswer:"

        # Tokenize and generate the response using the language model
        input_ids = self.llm.tokenizer.encode(combined_input, return_tensors="pt").to(self.llm.device)
        output = self.llm.generate(input_ids=input_ids, max_new_tokens=512, temperature=0.7, top_p=0.9)

        # Decode and return the generated answer
        return self.llm.tokenizer.decode(output[0], skip_special_tokens=True)



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
        num_documents = len(self.collection.list_documents())
        logger.info(f"Collection '{self.collection_name}' has {num_documents} documents.")
        if num_documents == 0:
            raise RuntimeError("ChromaDB collection is empty. Embeddings were not added successfully.")


def process_questions_in_batches(questions, embedding_model, chromadb_handler, llm, k=5, batch_size=8):
    """
    Process questions in batches to optimize memory usage and speed.

    Parameters:
        questions (list of str): List of questions to process.
        embedding_model (object): The model for generating question embeddings.
        chromadb_handler (ChromaDBHandler): Handler for querying the vector DB.
        llm (object): The language model for generating answers.
        k (int): Number of relevant documents to retrieve per question.
        batch_size (int): Number of questions to process in each batch.

    Returns:
        list of dict: Results with questions, retrieved documents, and answers.
    """
    results = []
    num_batches = math.ceil(len(questions) / batch_size)

    for i in range(num_batches):
        batch_questions = questions[i * batch_size:(i + 1) * batch_size]
        logger.info(f"Processing batch {i + 1}/{num_batches} with {len(batch_questions)} questions.")

        batch_embeddings = embedding_model.embed(batch_questions)  # Get embeddings in batch
        batch_results = []

        for embedding, question_text in zip(batch_embeddings, batch_questions):
            question = Question(text=question_text, embedding_model=embedding_model,
                                chromadb_handler=chromadb_handler, llm=llm, k=k)
            question.embedding = embedding.cpu().numpy()  # Use precomputed embedding
            retrieved_docs = question.retrieve_documents()
            answer = question.generate_answer()

            batch_results.append({
                "question": question_text,
                "retrieved_documents": retrieved_docs,
                "answer": answer
            })

        results.extend(batch_results)

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


def prepopulate_memory_db(memory_db, questions_answers, embedding_model):
    """
    Prepopulate MemoryDB with question-answer pairs.

    Parameters:
        memory_db (MemoryDB): The memory database instance.
        questions_answers (list of tuples): List of (question, answer) pairs.
        embedding_model (object): The model to generate embeddings.
    """
    for question, answer in questions_answers:
        # Generate embedding for the answer
        answer_embedding = embedding_model.embed(answer)

        # Add to memory DB with both the answer embedding and document (answer text)
        memory_db.add_to_memory(answer_embedding, answer)


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
    questions = dataframe[question_field].tolist()
    actual_doc_ids = dataframe[doc_id_field].tolist()
    answers, retrieved_contexts_list, retrieved_doc_ids_list = [], [], []

    if hasattr(embedding_model, embed_document_method):
        embed_func = getattr(embedding_model, embed_document_method)
        embeddings = embed_func(
            [f"{query_instruction}{text}" for text in questions] if query_instruction else questions)
    else:
        embeddings = embedding_model.embed(
            questions, batch_size=batch_size, instruction=query_instruction, max_length=max_length
        )
        embeddings = embeddings.cpu().numpy()

    results = chromadb_handler.query(query_embeddings=embeddings.tolist(), n_results=k)

    for idx in range(len(questions)):
        question = questions[idx]
        retrieved_documents = results["documents"][idx]
        retrieved_metadatas = results["metadatas"][idx]
        retrieved_doc_ids = [meta["doc_id"] for meta in retrieved_metadatas]
        retrieved_contexts = "\n\n".join(retrieved_documents)

        answer = llm.generate_response_with_context(
            context=retrieved_contexts, prompt=question, max_length=256, temperature=0.7, top_p=0.9
        )

        answers.append(answer)
        retrieved_contexts_list.append(retrieved_contexts)
        retrieved_doc_ids_list.append(retrieved_doc_ids)

    dataframe["Retrieved_Doc_IDs"] = retrieved_doc_ids_list
    dataframe["Retrieved_Contexts"] = retrieved_contexts_list
    dataframe["LLM_Answer"] = answers

    return dataframe


def run_evaluation(models, datasets, k):
    llm = HuggingFaceLLM(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
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

            df_with_answers = generate_answers(
                df, embedding_model, chromadb_handler, llm, k, context_field="context"
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

    # Create document embeddings
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

    memory_db = MemoryDB(embedding_dim=1024)

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

    # Example usage within the main function or script
    questions_answers = [
        ("Is Hirschsprung disease classified as a single-gene disorder or influenced by multiple factors?",
         "Coding sequence mutations in RET, GDNF, EDNRB, EDN3, and SOX10 are involved in the development of Hirschsprung disease. The majority of these genes was shown to be related to Mendelian syndromic forms of Hirschsprung's disease, whereas the non-Mendelian inheritance of sporadic non-syndromic Hirschsprung disease proved to be complex; involvement of multiple loci was demonstrated in a multiplicative model."),
        ("Identify signaling molecules that interact with the EGFR receptor.",
         "The 7 known EGFR ligands are: epidermal growth factor (EGF), betacellulin (BTC), epiregulin (EPR), heparin-binding EGF (HB-EGF), transforming growth factor-α [TGF-α], amphiregulin (AREG), and epigen (EPG)."),
        ("Do long non-coding RNAs get spliced?",
         "Long non coding RNAs appear to be spliced through the same pathway as the mRNAs"),
        ("Do cells release RANKL?",
         "Receptor activator of nuclear factor κB ligand (RANKL) is a cytokine predominantly secreted by osteoblasts."),
        ("What miRNAs show potential as biomarkers for epithelial ovarian cancer?",
         "miR-200a, miR-100, miR-141, miR-200b, miR-200c, miR-203, miR-510, miR-509-5p, miR-132, miR-26a, let-7b, miR-145, miR-182, miR-152, miR-148a, let-7a, let-7i, miR-21, miR-92 and miR-93 could be used as potential biomarkers for epithelial ovarian cancer."),
        ("What are the Yamanaka factors?",
         "The Yamanaka factors are the OCT4, SOX2, MYC, and KLF4 transcription factors"),
        ("Could the monoclonal antibody Trastuzumab (Herceptin) be beneficial in prostate cancer treatment?",
         "Although is still controversial, Trastuzumab (Herceptin) can be of potential use in the treatment of prostate cancer overexpressing HER2, either alone or in combination with other drugs.")
    ]

    prepopulate_memory_db(memory_db, questions_answers, embedding_model)

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
        print(f"{i}. Answer: {result['answer']}\n")