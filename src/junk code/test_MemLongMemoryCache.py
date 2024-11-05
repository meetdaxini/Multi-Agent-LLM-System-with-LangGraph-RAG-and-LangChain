import pandas as pd
import chromadb
import logging
import torch
import gc
import re
import os
from my_rag.components.embeddings.huggingface_embedding import HuggingFaceEmbedding
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from my_rag.components.llms.huggingface_llm import HuggingFaceLLM
from my_rag.components.memory_db.memlong_db import MemoryDB
from typing import List, Dict, Union, Tuple
from claud import LLMClient


DEFAULT_DATA_PATH = "data_test"  # Path to your PDFs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Question:
    def __init__(self, text, embedding_model_path, chromadb_handler, llm, k=2, chunk_size=1000, device=None):
        """
        Initialize a Question object.

        Parameters:
            text (str): The question text.
            embedding_model_path (str): Path or model name for HuggingFaceEmbedding.
            chromadb_handler (ChromaDBHandler): Handler for querying the vector DB.
            llm (HuggingFaceLLM): An instance of HuggingFaceLLM for generating answers.
            k (int): Number of relevant documents to retrieve.
            chunk_size (int): Number of tokens per chunk.
            device (Optional[str]): The device to load the embedding model on (e.g., 'cpu', 'cuda').
        """
        self.text = text
        self.embedder = HuggingFaceEmbedding(model_name=embedding_model_path, device=device)  # Initialize embedder
        self.chromadb_handler = chromadb_handler
        self.llm = llm
        self.k = k
        self.chunk_size = chunk_size
        self.embedding = None
        self.retrieved_documents = []
        self.answer = None

    def embed_question(self):
        """Generate embeddings for the question text."""
        # Use the embed method from HuggingFaceEmbedding to get normalized embeddings
        self.embedding = self.embedder.embed([self.text])[0].cpu().numpy()

    def retrieve_documents(self):
        """Query the vector DB to retrieve relevant documents."""
        if self.embedding is None:
            raise ValueError("Question embedding not generated. Call 'embed_question' first.")

        logger.info(f"Querying with embedding shape: {self.embedding.shape}")

        # Attempt to retrieve from MemoryDB first
        results = self.chromadb_handler.query(query_embeddings=[self.embedding], n_results=self.k)

        if "documents" not in results or not results["documents"][0]:
            print("No documents found in MemoryDB. Proceeding to fallback retrieval...")
            raise ValueError(
                "No documents retrieved. Make sure the vector database is populated and query embeddings are correct."
            )

        # Successfully found documents in MemoryDB
        self.retrieved_documents = results["documents"][0]
        print(f"Documents successfully retrieved from MemoryDB. Retrieved {len(self.retrieved_documents)} documents.")
        logger.info(f"Retrieved {len(self.retrieved_documents)} documents for the query.")

        return self.retrieved_documents

    def generate_answer(self, dataframe, context_field="context"):
        """
        Generate an answer using the LLM's internal prompt format, with retrieved documents as context.

        Parameters:
            dataframe (pd.DataFrame): DataFrame containing retrieved documents and context text.
            context_field (str): Column name for the context text in the DataFrame.
        """
        if dataframe.empty:
            raise ValueError("Dataframe is empty. No context available to generate an answer.")

        # Combine relevant context texts
        context = "\n\n".join(dataframe[context_field].tolist())
        prompt = self.text

        # Construct a prompt with the question text and the retrieved context
        # prompt = f"Context:\n{context}\n\nQuestion: {self.text}\n\nAnswer:"

        try:
            # Pass only the context to the LLM and rely on its internal prompt structure
            response = self.llm.generate_response_with_context(
                context=context, max_length=200, temperature=0.7, top_p=0.9, prompt=prompt)
            self.answer = response.strip()
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            self.answer = "Could not generate a relevant answer."

        return self.answer


class MemoryAugmentedRAG:
    def __init__(self, embedding_model_path, embedding_dim, llm, max_memory_size=10000, k=2, similarity_threshold=0.4):
        # Load embedding model for chunk embeddings
        self.embedder = HuggingFaceEmbedding(embedding_model_path)
        self.memory_db = MemoryDB(embedding_dim=embedding_dim, max_memory_size=max_memory_size, similarity_threshold=similarity_threshold)
        self.k = k
        self.llm = llm  # Pass the language model instance to the constructor

    def embed_question(self, text):
        """Embed the input question for retrieval."""
        return self.embedder.embed([text])

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
            answer_embedding = self.embedder.embed([answer])[0]

            # Add to memory DB
            self.memory_db.add_to_memory(answer_embedding, answer)

    # def generate_answer(self, question_text):
    #     """Generate an answer using the retrieved documents as context."""
    #     # Embed the query
    #     query_embedding = self.embed_question(question_text)
#
    #     # Retrieve relevant documents
    #     retrieved_docs = self.retrieve_documents(query_embedding)
#
    #     # Combine the documents into a single context
    #     context = "\n\n".join(retrieved_docs)
    #     combined_input = f"{context}\n\nQuestion: {question_text}\n\nAnswer:"
#
    #     # Tokenize and generate the response using the language model
    #     input_ids = self.llm.tokenizer.encode(combined_input, return_tensors="pt").to(self.llm.device)
    #     output = self.llm.generate(input_ids=input_ids, max_new_tokens=512, temperature=0.4, top_p=0.9, top_k=10)
    #     print("THIS WAS GENERATED BY memlong")
#
    #     # Decode and return the generated answer
    #     return self.llm.tokenizer.decode(output[0], skip_special_tokens=True)


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


def max_similarity(retrieved_docs, query_embedding):
    # Convert documents' embeddings to tensor
    doc_embeddings = torch.stack([doc[0] for doc in retrieved_docs])  # retrieved_docs is a list of (embedding, document)
    # Normalize for cosine similarity
    query_norm = query_embedding / torch.norm(query_embedding)
    doc_norms = doc_embeddings / torch.norm(doc_embeddings, dim=1, keepdim=True)
    # Compute cosine similarities
    similarities = torch.matmul(query_norm, doc_norms.T)
    # Return the maximum similarity score
    return similarities.max().item()


# def process_questions_in_batches(
#         questions, embedding_model, memory_db, chromadb_handler, llm, k=2, batch_size=8, similarity_threshold=0.8
# ):
#     results = []
#     num_batches = math.ceil(len(questions) / batch_size)
#
#     for i in range(num_batches):
#         batch_questions = questions[i * batch_size:(i + 1) * batch_size]
#
#         # Generate embeddings in batch
#         batch_embeddings = embedding_model.embed(batch_questions)
#
#         batch_results = []
#         for embedding, question_text in zip(batch_embeddings, batch_questions):
#             # 1. Try retrieving from MemoryDB first
#             retrieved_docs = memory_db.retrieve_top_k(embedding, k=k)
#             retrieval_source = "MemoryDB"
#
#             # Check if MemoryDB retrieval meets similarity threshold
#             if not retrieved_docs or max_similarity(retrieved_docs, embedding) < similarity_threshold:
#                 # 2. Fallback to ChromaDB if MemoryDB retrieval is insufficient
#                 chroma_docs = chromadb_handler.query([embedding.cpu().numpy()], n_results=k)
#                 retrieved_docs = [doc for sublist in chroma_docs.get("documents", []) for doc in
#                                   (sublist if isinstance(sublist, list) else [sublist])]
#                 retrieval_source = "ChromaDB"
#                 print("No similar question found in MemoryDB. Processing with ChromaDB...(Main test File)")
#
#             # Combine retrieved documents into context for LLM generation
#             context = "\n\n".join(retrieved_docs)
#             combined_input = f"{context}\n\nQuestion: {question_text}\n\nAnswer:"
#
#             # Prepare input for the model
#             input_ids = llm.tokenizer(
#                 combined_input,
#                 return_tensors="pt",
#                 truncation=True,
#                 max_length=2048  # Adjust based on your model's context window
#             ).input_ids.to(llm.device)
#
#             # Generate response using LLM with specified parameters
#             try:
#                 with torch.no_grad():
#                     output_ids = llm.model.generate(
#                         input_ids=input_ids,
#                         max_new_tokens=250,
#                         temperature=0.2,
#                         top_p=0.9,
#                         top_k=10,
#                         num_beams=3,
#                         pad_token_id=llm.tokenizer.eos_token_id,
#                         do_sample=True
#                     )
#
#                 # Decode only the newly generated tokens
#                 answer = llm.tokenizer.decode(
#                     output_ids[0][input_ids.shape[1]:],
#                     skip_special_tokens=True,
#                     clean_up_tokenization_spaces=True
#                 ).strip()
#
#             except Exception as e:
#                 logger.error(f"Error generating answer: {e}")
#                 answer = "An error occurred while generating the answer."
#
#             batch_results.append({
#                 "question": question_text,
#                 "answer": answer,
#                 "retrieval_source": retrieval_source
#             })
#
#         results.extend(batch_results)
#
#         # Clean up GPU memory
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#         gc.collect()
#
#     return results


def alphanumeric_string(input_string):
    return re.sub(r"[^a-zA-Z0-9]", "", input_string)


def log_cuda_memory_usage(message=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1536 ** 2)
        reserved = torch.cuda.memory_reserved() / (1536 ** 2)
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


def prepopulate_memory_db(memory_db, questions_answers: List[Tuple[str, str]], embedding_model: HuggingFaceEmbedding):
    """
    Prepopulate MemoryDB with question-answer pairs.

    Parameters:
        memory_db (MemoryDB): The memory database instance.
        questions_answers (list of tuples): List of (question, answer) pairs.
        embedding_model (HuggingFaceEmbedding): The embedding model to generate embeddings.
    """
    for question, answer in questions_answers:
        # Generate embedding for the answer (pass as a list to embed method)
        answer_embedding = embedding_model.embed([answer])[0]  # Retrieves the embedding tensor for the answer text

        # Add to memory DB with both the answer embedding and document (answer text)
        memory_db.add_to_memory(answer_embedding, answer)


# def run_evaluation(models, datasets, k):
#     llm = HuggingFaceLLM(
#         model_name="meta-llama/Meta-Llama-3-8B-Instruct",
#         device="cuda" if torch.cuda.is_available() else "cpu",
#         torch_dtype=torch.float16,
#         load_in_8bit=True,
#         device_map="auto",
#         trust_remote_code=True,
#     )
#
#     for dataset_info in datasets:
#         df = load_dataset(dataset_info["path"])
#         for model_info in models:
#             model_name = model_info["name"]
#             embedding_model = HuggingFaceEmbedding(
#                 model_name=model_name, load_in_8bit=model_info.get("load_in_8bit", False), device_map="auto"
#             )
#
#             chunked_texts, document_ids, context_embeddings = create_document_embeddings(
#                 embedding_model, df, chunk_size=1000, chunk_overlap=115
#             )
#
#             ids = [f"{doc_id}_{idx}" for idx, doc_id in enumerate(document_ids)]
#             metadatas = [{"doc_id": doc_id} for doc_id in document_ids]
#
#             chromadb_handler = ChromaDBHandler(collection_name=alphanumeric_string(model_name))
#             chromadb_handler.add_embeddings(
#                 embeddings=context_embeddings, documents=chunked_texts, metadatas=metadatas, ids=ids
#             )
#
#             # Prepare a list to collect answers
#             answers = []
#
#             for _, row in df.iterrows():
#                 # Create a Question instance for each question
#                 question_text = row["question"]
#                 question_instance = Question(
#                     text=question_text,
#                     embedding_model=embedding_model,
#                     chromadb_handler=chromadb_handler,
#                     llm=llm,
#                     k=k,
#                 )
#
#                 # Step 1: Generate the question embedding and retrieve documents
#                 question_instance.embed_question()
#                 question_instance.retrieve_documents()
#
#                 # Step 2: Generate an answer without additional arguments
#                 answer = question_instance.generate_answer()
#                 answers.append({
#                     "question": question_text,
#                     "answer": answer,
#                     "retrieved_documents": question_instance.retrieved_documents
#                 })
#
#             # Convert answers to DataFrame and save as Excel
#             df_with_answers = pd.DataFrame(answers)
#             output_filename = f"retriever_eval_{model_name.replace('/', '_')}.xlsx"
#             df_with_answers.to_excel(output_filename, index=False)
#             logger.info(f"Results saved to {output_filename}")
#
#             # Cleanup
#             chromadb_handler.delete_collection()
#
#     llm.clean_up()


if __name__ == "__main__":
    # Load your dataset
    df = load_dataset()

    # Initialize your embedding model
    embedding_model = HuggingFaceEmbedding(model_name="dunzhang/stella_en_1.5B_v5")

    # Initialize ChromaDB handler and MemoryDB
    chromadb_handler = ChromaDBHandler(collection_name="my_collection")
    memory_db = MemoryDB(embedding_dim=1536)  # Ensure this matches your model’s embedding dimension

    # Add embeddings to ChromaDB if not already done
    if chromadb_handler.document_count == 0:
        # Prepopulate ChromaDB with document embeddings
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

        chromadb_handler.add_embeddings(
            embeddings=context_embeddings,
            documents=chunked_texts,
            metadatas=metadatas,
            ids=ids
        )

    # Prepopulate MemoryDB with relevant question-answer pairs
    questions_answers = [
        ("Is Hirschsprung disease classified as a single-gene disorder or influenced by multiple factors?",
         "Coding sequence mutations in RET, GDNF, EDNRB, EDN3, and SOX10 are involved in the development of Hirschsprung disease. The majority of these genes was shown to be related to Mendelian syndromic forms of Hirschsprung's disease, whereas the non-Mendelian inheritance of sporadic non-syndromic Hirschsprung disease proved to be complex; involvement of multiple loci was demonstrated in a multiplicative model."),
        ("Identify signaling molecules that interact with the EGFR receptor.",
         "The 7 known EGFR ligands are: epidermal growth factor (EGF), betacellulin (BTC), epiregulin (EPR), heparin-binding EGF (HB-EGF), transforming growth factor-α [TGF-α], amphiregulin (AREG), and epigen (EPG)."),
        ("Do long non-coding RNAs get spliced?",
         "Long non coding RNAs appear to be spliced through the same pathway as the mRNAs"),
        # ("Do cells release RANKL?",
        #  "Receptor activator of nuclear factor κB ligand (RANKL) is a cytokine predominantly secreted by osteoblasts."),
        # ("What miRNAs show potential as biomarkers for epithelial ovarian cancer?",
        #  "miR-200a, miR-100, miR-141, miR-200b, miR-200c, miR-203, miR-510, miR-509-5p, miR-132, miR-26a, let-7b, miR-145, miR-182, miR-152, miR-148a, let-7a, let-7i, miR-21, miR-92 and miR-93 could be used as potential biomarkers for epithelial ovarian cancer."),
        # ("What are the Yamanaka factors?",
        #  "The Yamanaka factors are the OCT4, SOX2, MYC, and KLF4 transcription factors"),
        # ("Could the monoclonal antibody Trastuzumab (Herceptin) be beneficial in prostate cancer treatment?",
        #  "Although is still controversial, Trastuzumab (Herceptin) can be of potential use in the treatment of prostate cancer overexpressing HER2, either alone or in combination with other drugs.")
    ]

    prepopulate_memory_db(memory_db, questions_answers, embedding_model)
    # TODO: Add more supported models here
    #  Yes, the embedding.shape for the Question class should be compatible with the embedding_dim specified for MemoryDB.
    #  Here’s why:
    #  Embedding Consistency:
    #       The embeddings generated by HuggingFaceEmbedding in the Question class (i.e., self.embedding) need to have
    #       the same dimensionality as those stored in MemoryDB. This ensures compatibility during retrieval operations
    #       (such as similarity calculations) within the MemoryDB class.
    #  Matching Embedding Dimensions: If MemoryDB is initialized with embedding_dim=1536, then the embeddings generated
    #   by the model in the Question class should also have a shape of (1536,) for each question embedding.

    # Define your LLM
    llm = HuggingFaceLLM(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float16,
        load_in_8bit=True,
        device_map="auto",
        trust_remote_code=True,
    )

    # Select one question to process
    # TODO: Change question one at a time for testing purposes
    question_text = "Do cells release RANKL?"

    # Create a Question instance for the selected question
    question_instance = Question(
        text=question_text,
        embedding_model_path="dunzhang/stella_en_1.5B_v5",
        chromadb_handler=chromadb_handler,
        llm=llm,
        k=5,
    )

    # Step 1: Generate the question embedding and retrieve documents
    question_instance.embed_question()
    retrieved_docs = question_instance.retrieve_documents()

    # Step 2: Generate an answer using the retrieved documents as context
    if retrieved_docs:
        df_retrieved = pd.DataFrame({"context": retrieved_docs})
        answer = question_instance.generate_answer(dataframe=df_retrieved)
        print(f"Question: {question_text}")
        print(f"Answer: {answer}\n")
    else:
        print("No documents retrieved. Unable to generate an answer.")



