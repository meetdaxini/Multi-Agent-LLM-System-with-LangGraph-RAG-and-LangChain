import os
import gc
import json
import torch
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import logging
from logging.handlers import RotatingFileHandler
from abc import ABC, abstractmethod
from my_rag.components.embeddings.huggingface_embedding import HuggingFaceEmbedding
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from my_rag.components.memory_db.junk_memo_rag import MemoryDB
from my_rag.components.llms.huggingface_llm import HuggingFaceLLM
import chromadb

DEFAULT_DATA_PATH = "data_test"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type aliases for clarity
DocumentID = str
EmbeddingVector = List[float]
Context = str


@dataclass
class Document:
    """Represents a document with its content and metadata."""
    content: str
    metadata: Dict[str, Any]
    id: str


class VectorStore(ABC):
    """Abstract base class for vector stores."""
    @abstractmethod
    def add_documents(self, documents: List[str], embeddings: List[EmbeddingVector], metadatas: List[Dict[str, Any]],
                      ids: List[str]) -> None:
        pass

    @abstractmethod
    def query(self, query_embeddings: List[EmbeddingVector], n_results: int, include: Optional[List[str]] = None) -> \
    Dict[str, Any]:
        pass

    @abstractmethod
    def delete_collection(self) -> None:
        pass

    @abstractmethod
    def verify_population(self) -> None:
        pass


class ChromaDBStore(VectorStore):
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.client = chromadb.Client()  # Assuming an in-memory ChromaDB client for simplicity

        # Try to delete any pre-existing collection with the same name
        try:
            self.client.delete_collection(name=collection_name)
            logger.info(f"Deleted existing collection: {collection_name}")
        except Exception as e:
            logger.info(f"No existing collection to delete. Starting fresh: {e}")

        # Create the new collection and initialize document count
        self.collection = self.client.create_collection(name=collection_name)
        self.document_count = 0
        logger.info(f"Created ChromaDB collection (database) with name: {self.collection_name}")

    def add_documents(self, documents: List[str], embeddings: List[EmbeddingVector], metadatas: List[Dict[str, Any]],
                      ids: List[str]) -> None:
        """Add embeddings along with associated documents, metadata, and ids to the ChromaDB collection."""
        if embeddings is None or len(embeddings) == 0:
            raise ValueError("No embeddings to add to ChromaDB.")

        if not (len(embeddings) == len(documents) == len(metadatas) == len(ids)):
            raise ValueError("Embeddings, documents, metadatas, and ids must all have the same length.")

        # Add embeddings and documents to the collection
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        # Update and log document count
        self.document_count += len(embeddings)
        logger.info(
            f"Added {len(embeddings)} embeddings to the ChromaDB collection '{self.collection_name}'. Total documents: {self.document_count}")

    def query(self, query_embeddings: List[EmbeddingVector], n_results: int, include: Optional[List[str]] = None) -> \
    Dict[str, Any]:
        """Query the collection with provided embeddings and retrieve top results."""
        if include is None:
            include = ["documents", "metadatas"]
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            include=include,
        )
        return results

    def delete_collection(self) -> None:
        """Delete the collection and reset document count."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.document_count = 0
            logger.info(f"Deleted collection '{self.collection_name}' and reset document count.")
        except Exception as e:
            logger.error(f"Error deleting collection '{self.collection_name}': {e}")

    def verify_population(self) -> None:
        """Verify if the collection has documents and log the count."""
        logger.info(f"Collection '{self.collection_name}' currently holds {self.document_count} documents.")
        if self.document_count == 0:
            raise RuntimeError("ChromaDB collection is empty. Embeddings were not added successfully.")
        else:
            logger.info("ChromaDB collection populated successfully.")

    def get_document_count(self) -> int:
        """Return the current document count."""
        return self.document_count

class DocumentProcessor:
    """Handles document loading and preprocessing."""
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 115):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_and_split_documents(self, directory: Path) -> List[Document]:
        """Load PDFs from directory and split into chunks."""
        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory}")
        loader = PyPDFDirectoryLoader(str(directory))
        documents = loader.load()
        processed_documents: List[Document] = []
        for doc in documents:
            chunks = self.text_splitter.split_text(doc.page_content)
            source = doc.metadata.get("source", "")
            processed_documents.extend([
                Document(
                    content=chunk,
                    metadata={"source": source},
                    id=f"{source}_{i}"
                )
                for i, chunk in enumerate(chunks)
            ])
        return processed_documents


def log_cuda_memory_usage(message=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1536 ** 2)
        reserved = torch.cuda.memory_reserved() / (1536 ** 2)
        logger.info(f"{message} - CUDA memory allocated: {allocated:.2f} MB, reserved: {reserved:.2f} MB")
    else:
        logger.info(f"{message} - CUDA not available")


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


class RAGSystem:
    """Main RAG system orchestrating components."""
    def __init__(
            self,
            embedding_model: HuggingFaceEmbedding,
            vector_store: ChromaDBStore,
            memory_bank: MemoryDB,
            llm: HuggingFaceLLM,
            logger: Optional[logging.Logger] = None
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.memory_bank = memory_bank
        self.llm = llm
        self.logger = logger or self._setup_default_logger()

    @staticmethod
    def _setup_default_logger() -> logging.Logger:
        """Create a default logger with file rotation."""
        logger = logging.getLogger("RAGSystem")
        logger.setLevel(logging.INFO)
        handler = RotatingFileHandler(
            "rag_system.log",
            maxBytes=1536 * 1536,  # 1MB
            backupCount=5
        )
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def index_documents(self, documents: List[Document]) -> None:
        """Index documents into the vector store."""
        dataframe = pd.DataFrame({
            "context": [doc.content for doc in documents],
            "source_doc": [doc.id for doc in documents]
        })

        try:
            # Generate document embeddings and prepare chunks
            chunked_texts, chunked_doc_ids, embeddings = create_document_embeddings(
                embedding_model=self.embedding_model,
                dataframe=dataframe,
                context_field="context",
                doc_id_field="source_doc"
            )

            # Create Document objects for each chunk with unique IDs
            chunked_documents = [
                Document(content=text, metadata={"source": doc_id}, id=f"{doc_id}_{idx}")
                for idx, (text, doc_id) in enumerate(zip(chunked_texts, chunked_doc_ids))
            ]

            # Add the documents and their embeddings to the vector store
            self.vector_store.add_documents(
                documents=[doc.content for doc in chunked_documents],
                embeddings=embeddings,
                metadatas=[doc.metadata for doc in chunked_documents],
                ids=[doc.id for doc in chunked_documents]
            )

            self.logger.info(f"Indexed {len(chunked_documents)} document chunks successfully in '{self.vector_store.collection_name}'")
        except Exception as e:
            self.logger.error(f"Failed to index documents: {str(e)}")
            raise

    def prepopulate_memory_bank(self, questions_answers: List[Tuple[str, str]]) -> None:
        """Populate memory bank with question-answer pairs."""
        for question, answer in questions_answers:
            answer_embedding = self.embedding_model.embed([answer])[0]
            self.memory_bank.add_to_memory(answer_embedding, answer)
            self.logger.info(f"Populated memory with answer: {answer[:50]}...")

    def query(self, question: str, n_results: int = 3) -> str:
        """Process a question and return an answer."""
        try:
            # Step 1: Embed the question for retrieval
            question_embedding = self.embedding_model.embed([question])[0]

            # Step 2: Attempt to retrieve from MemoryDB
            memory_results, similarity_scores = self.memory_bank.retrieve_top_k(question_embedding, k=n_results)

            if memory_results and all(score >= self.memory_bank.similarity_threshold for score in similarity_scores):
                context = memory_results[0]  # Only the top relevant answer
                self.logger.info("Retrieved context from MemoryDB.")
            else:
                # Fallback to ChromaDBStore
                results = self.vector_store.query(
                    query_embeddings=[question_embedding.tolist()],
                    n_results=n_results
                )

                relevant_docs = results.get("documents", [])
                if not relevant_docs:
                    self.logger.warning("No relevant documents found for query.")
                    return "No relevant information found to answer the question."

                # Combine document contents for context
                context = "\n\n".join(relevant_docs[0])
                self.logger.info("Retrieved context from ChromaDBStore.")

            # Step 4: Generate answer using LLM
            answer = self.llm.generate_response_with_context(
                context=context,
                prompt=question,
                max_new_tokens=250
            )

            self.logger.info(f"Successfully generated answer for question: {question[:50]}...")
            return answer

        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            raise


def main():

    embedding_model = HuggingFaceEmbedding(
        model_name="dunzhang/stella_en_1.5B_v5"
    )
    vector_store = ChromaDBStore(collection_name="main_data_bank")
    memory_bank = MemoryDB(similarity_threshold=0.29, max_memory_size=1000, compression_ratio=8)
    llm = HuggingFaceLLM(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float16,
        load_in_8bit=True,
        device_map="auto",
        trust_remote_code=True
    )

    rag_system = RAGSystem(
        embedding_model=embedding_model,
        vector_store=vector_store,
        memory_bank=memory_bank,
        llm=llm
    )

    # Process documents
    doc_processor = DocumentProcessor()
    documents = doc_processor.load_and_split_documents(Path(DEFAULT_DATA_PATH))
    rag_system.index_documents(documents)

    # Define question-answer pairs for memory bank pre-population
    questions_answers = [
        ("Question: Is Hirschsprung disease classified as a single-gene disorder or influenced by multiple factors?",
         "Answer: Coding sequence mutations in RET, GDNF, EDNRB, EDN3, and SOX10 are involved in the development of Hirschsprung disease. The majority of these genes was shown to be related to Mendelian syndromic forms of Hirschsprung's disease, whereas the non-Mendelian inheritance of sporadic non-syndromic Hirschsprung disease proved to be complex; involvement of multiple loci was demonstrated in a multiplicative model."),
        ("Question: Identify signaling molecules that interact with the EGFR receptor.",
         "Answer: The 7 known EGFR ligands are: epidermal growth factor (EGF), betacellulin (BTC), epiregulin (EPR), heparin-binding EGF (HB-EGF), transforming growth factor-α [TGF-α], amphiregulin (AREG), and epigen (EPG)."),
        ("Question: Do long non-coding RNAs get spliced?",
         "Answer: Long non coding RNAs appear to be spliced through the same pathway as the mRNAs"),
        ("Do cells release RANKL?",
         "Receptor activator of nuclear factor κB ligand (RANKL) is a cytokine predominantly secreted by osteoblasts."),
        # ("What miRNAs show potential as biomarkers for epithelial ovarian cancer?",
        #  "miR-200a, miR-100, miR-141, miR-200b, miR-200c, miR-203, miR-510, miR-509-5p, miR-132, miR-26a, let-7b, miR-145, miR-182, miR-152, miR-148a, let-7a, let-7i, miR-21, miR-92 and miR-93 could be used as potential biomarkers for epithelial ovarian cancer."),
        # ("What are the Yamanaka factors?",
        #  "The Yamanaka factors are the OCT4, SOX2, MYC, and KLF4 transcription factors"),
        # ("Could the monoclonal antibody Trastuzumab (Herceptin) be beneficial in prostate cancer treatment?",
        #  "Although is still controversial, Trastuzumab (Herceptin) can be of potential use in the treatment of prostate cancer overexpressing HER2, either alone or in combination with other drugs.")
    ]

    # Prepopulate memory bank
    rag_system.prepopulate_memory_bank(questions_answers)

    # Example list of questions
    questions = [
        "Is Hirschsprung disease a Mendelian or a multifactorial disorder?",
        "List signaling molecules (ligands) that interact with the receptor EGFR?",
        "Are long non-coding RNAs spliced?",
        "Is RANKL secreted from the cells?",
        "Which miRNAs could be used as potential biomarkers for epithelial ovarian cancer?",
        "Which are the Yamanaka factors?",
        "Is the monoclonal antibody Trastuzumab (Herceptin) of potential use in the treatment of prostate cancer?"
    ]

    # Dictionary to store question-answer pairs
    answers = {}

    # Process each question, print, and store the answer
    for question in questions:
        answer = rag_system.query(question)
        answers[question] = answer  # Store the question-answer pair in the dictionary

        # Print the question and answer
        print(f"\n\n❀❀ Question ❀❀: {question}")
        print(f"❀❀ Answer ❀❀: {answer}")

    # Save the results in JSON format
    output_file = "question_answers.json"
    with open(output_file, "w") as file:
        json.dump(answers, file, indent=4)

    print(f"\nResults saved to {output_file}")

    # # Prepopulate memory bank
    # rag_system.prepopulate_memory_bank(questions_answers)
    #
    # # Example query
    # question = "Is Hirschsprung disease classified as a single-gene disorder or influenced by multiple factors?"
    # answer = rag_system.query(question)
    # print(f"\n\n❀❀ Question ❀❀: {question}")
    # print(f"❀❀ Answer ❀❀: {answer}")


if __name__ == "__main__":
    main()



import json

# Example list of questions
# questions = [
#     "Is Hirschsprung disease classified as a single-gene disorder or influenced by multiple factors?",
#     "What are the Yamanaka factors?",
#     "Can long non-coding RNAs be spliced?",
#     "Identify signaling molecules that interact with the EGFR receptor."
# ]
#
# # Dictionary to store question-answer pairs
# answers = {}
#
# # Process each question and store the answer
# for question in questions:
#     answer = rag_system.query(question)
#     answers[question] = answer  # Store the question-answer pair in the dictionary
#
# # Save the results in JSON format
# output_file = "question_answers.json"
# with open(output_file, "w") as file:
#     json.dump(answers, file, indent=4)
#
# print(f"Results saved to {output_file}")


