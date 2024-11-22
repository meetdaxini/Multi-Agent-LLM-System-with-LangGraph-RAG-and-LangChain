from my_rag.components.embeddings.huggingface_embedding import HuggingFaceEmbedding
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from my_rag.components.llms.huggingface_llm import HuggingFaceLLM
from typing import List, Optional, Dict, Any, Tuple, Union
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from logging.handlers import RotatingFileHandler
from sentence_transformers import CrossEncoder
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from rank_bm25 import BM25Okapi
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import chromadb
import logging
import pickle
import torch
import json
import zlib
import gc
import re
import os


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


# Configuration
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


# Define dataclasses and type aliases
@dataclass
class PromptTemplate:
    """Template for structured prompt generation."""
    template: str
    required_fields: List[str]

    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)


@dataclass
class RAPTORNode:
    """Represents a node in the RAPTOR hierarchy."""
    content: str
    summary: str
    embedding: np.ndarray
    level: int
    node_id: str
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    relevance_score: float = 0.0

    def __lt__(self, other):
        return self.relevance_score > other.relevance_score


@dataclass
class QueryResult:
    """Stores query results and metadata."""
    query: str
    transformed_query: str
    context: str
    response: str
    confidence: float
    feedback_score: Optional[float] = None


# VectorStore implementation
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


class RelevanceScorer:
    """Refined multi-level relevance scoring system."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def score_relevance(self, query: str, passages: List[str]) -> List[float]:
        """Score relevance of passages at multiple levels, prioritize deeper context if relevant."""
        scores = self.model.predict([[query, passage] for passage in passages])

        # Apply weighting to prioritize relevant multi-level context
        depth_weighted_scores = [score * (0.9 ** depth) for depth, score in enumerate(scores)]
        return depth_weighted_scores

    def rerank_results(self, results: List[Tuple[RAPTORNode, float]]) -> List[RAPTORNode]:
        """Rerank nodes by relevance, ensuring context-sensitive retrieval."""
        # Sort results by relevance score (the float value in the tuple)
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        # Extract only the RAPTORNode objects from the sorted list
        return [node for node, _ in sorted_results]


class SemanticAnalyzer:
    """Enhanced semantic analysis and coherence verification using transformer-based models."""

    def __init__(self, embedding_model_sentece: SentenceTransformer, coherence_threshold: float = 0.7):
        self.embedding_model = embedding_model_sentece
        self.coherence_threshold = coherence_threshold

    def compute_semantic_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        return cosine_similarity([embedding1], [embedding2])[0][0]

    def verify_semantic_coherence(self, parent_node: RAPTORNode, child_nodes: List[RAPTORNode]) -> bool:
        """Use transformer embeddings to verify coherence between parent and child nodes."""
        parent_embedding = self.embedding_model.encode(parent_node.content)
        child_embeddings = np.stack([self.embedding_model.encode(node.content) for node in child_nodes])
        mean_child_embedding = np.mean(child_embeddings, axis=0)

        return self.compute_semantic_similarity(parent_embedding, mean_child_embedding) >= self.coherence_threshold

    def find_relevant_nodes(self, query: str, nodes: List[RAPTORNode]) -> List[RAPTORNode]:
        """Find and return nodes relevant to the query based on transformer-based semantic similarity."""
        query_embedding = self.embedding_model.encode(query)
        relevant_nodes = [
            node for node in nodes
            if self.compute_semantic_similarity(query_embedding, node.embedding) > self.coherence_threshold
        ]
        return sorted(relevant_nodes, key=lambda x: x.relevance_score, reverse=True)


class RAPTORTree:
    """Manages the hierarchical tree structure of documents with dynamic routing."""

    def __init__(self):
        self.nodes: Dict[str, RAPTORNode] = {}

    def add_node(self, node: RAPTORNode):
        self.nodes[node.node_id] = node
        if node.parent_id:
            parent = self.nodes.get(node.parent_id)
            if parent:
                parent.children_ids.append(node.node_id)

    def get_top_nodes(self) -> List[RAPTORNode]:
        """Retrieve top-level nodes for the starting point in the tree."""
        return [node for node in self.nodes.values() if node.level == 1]

    def get_children(self, node: RAPTORNode) -> List[RAPTORNode]:
        """Retrieve child nodes of a given node."""
        return [self.nodes[child_id] for child_id in node.children_ids if child_id in self.nodes]

    def hierarchical_query(self, question: str, analyzer: SemanticAnalyzer, scorer: RelevanceScorer) -> str:
        """Query the RAPTOR tree hierarchically, dynamically routing through relevant nodes."""
        current_nodes = self.get_top_nodes()
        relevant_contexts = []

        while current_nodes:
            relevant_nodes = analyzer.find_relevant_nodes(question, current_nodes)
            if not relevant_nodes:
                break
            current_nodes = [child for node in relevant_nodes for child in self.get_children(node)]
            relevant_contexts.extend([node.content for node in relevant_nodes])

        # Aggregate context from relevant nodes
        refined_context = "\n".join(relevant_contexts[:2])  # Limit for context length

        return refined_context


class PromptTransformer:
    """Handles prompt transformation and refinement."""

    def __init__(self, llm: HuggingFaceLLM):
        self.llm = llm
        self.history: List[QueryResult] = []

    def transform_prompt(self, original_query: str, context: Optional[str] = None) -> str:
        """Transform the original query using HuggingFaceLLM for more effective prompting."""
        # Construct the input with optional context
        input_text = f"Refine query: {original_query}\nContext: {context}" if context else original_query

        # Use HuggingFaceLLM to transform the prompt
        transformed_query = self.llm.generate_response_with_context(context=input_text, prompt=input_text, max_new_tokens=128)
        return transformed_query

    def generate_abstract(self, content: str) -> str:
        """Generate an abstract or summary for high-level nodes."""
        prompt = f"Summarize this content for a high-level overview:\n\n{content}"
        summary = self.llm.generate_response_with_context(context="", prompt=prompt, max_new_tokens=150)
        return summary

    def learn_from_feedback(self, query_result: QueryResult):
        """Learn from query results and feedback."""
        self.history.append(query_result)

def safe_mean(scores):
    """Compute the mean of a list safely, handling empty lists and NaN values."""
    scores = [score for score in scores if not np.isnan(score)]  # Filter out NaN values
    if scores:
        return np.mean(scores)  # Calculate mean if the list is not empty
    return 0.0  # Default value if list is empty or all values were NaN


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

    def query(self, query_embeddings: List[List[float]], n_results: int, include: Optional[List[str]] = None) -> dict:
        """Query with provided embeddings and retrieve top results."""
        if include is None:
            include = ["documents", "metadatas", "distances"]

        try:
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                include=include,
            )
            # Ensure results have default structure if fields are missing
            return {
                "documents": results.get("documents", [[]]),
                "metadatas": results.get("metadatas", [[]]),
                "distances": results.get("distances", [[]]),
            }
        except Exception as e:
            logger.error(f"Error querying vector store: {str(e)}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

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


def decompress_embedding(compressed_data: bytes) -> np.ndarray:
    """Decompress a single embedding vector."""
    return pickle.loads(zlib.decompress(compressed_data)).astype(np.float32)


class StorageOptimizer:
    """Handles embedding compression and storage optimization."""

    def __init__(self, dimension_threshold: int = 64, compression_level: int = 9):
        self.scaler = StandardScaler()
        self.dimension_threshold = dimension_threshold
        self.compression_level = compression_level

    def compress_embedding(self, embedding: np.ndarray) -> bytes:
        """Compress a single embedding vector."""
        # Convert to half precision (float16)
        compressed = embedding.astype(np.float16)
        # Serialize and compress
        return zlib.compress(pickle.dumps(compressed), level=self.compression_level)

    def quantize_embedding(self, embedding: np.ndarray, bits: int = 8) -> np.ndarray:
        """Quantize embedding to reduce precision with NaN handling."""
        # Ensure that NaN values are replaced with 0
        embedding = np.nan_to_num(embedding, nan=0.0)  # Explicitly set NaNs to 0.0

        if bits == 8:
            # Normalize to [-1, 1] range before scaling to [0, 255]
            normalized = np.clip(embedding, -1, 1)
            return np.clip(normalized * 127.5 + 127.5, 0, 255).astype(np.uint8)
        elif bits == 4:
            normalized = np.clip(embedding, -1, 1)
            return np.clip(normalized * 7.5 + 7.5, 0, 15).astype(np.uint8)
        return embedding

    def dequantize_embedding(self, quantized: np.ndarray, bits: int = 8) -> np.ndarray:
        """Dequantize embedding back to original scale."""
        if bits == 8:
            return (quantized.astype(np.float32) - 128) / 128
        elif bits == 4:
            return (quantized.astype(np.float32) - 8) / 8
        return quantized


class OptimizedChromaDBStore(ChromaDBStore):
    """Storage-optimized version of ChromaDBStore."""

    def __init__(self, collection_name: str, storage_path: str, max_documents: Optional[int] = None):
        super().__init__(collection_name)
        self.storage_path = storage_path
        self.storage_optimizer = StorageOptimizer()
        self.max_documents = max_documents
        self.embedding_cache = {}

        # Create storage directory if it doesn't exist
        os.makedirs(storage_path, exist_ok=True)

        # Initialize disk-based client instead of in-memory
        self.client = chromadb.PersistentClient(path=storage_path)

    def add_documents(self, documents: List[str], embeddings: List[List[float]], metadatas: List[dict],
                      ids: List[str]) -> None:
        """Add documents with length validation and error handling."""
        # Validate input lengths
        lengths = {
            'documents': len(documents),
            'embeddings': len(embeddings),
            'metadatas': len(metadatas),
            'ids': len(ids)
        }

        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"Inconsistent lengths detected: {lengths}. All inputs must have the same length."
            )

        # Ensure the storage path exists
        os.makedirs(self.storage_path, exist_ok=True)
        data_path = self.storage_path
        os.makedirs(data_path, exist_ok=True)

        try:
            # Compress embeddings before storage
            compressed_embeddings = []
            for emb in embeddings:
                emb_array = np.array(emb)
                quantized = self.storage_optimizer.quantize_embedding(emb_array)
                compressed = self.storage_optimizer.compress_embedding(quantized)
                compressed_embeddings.append(compressed)

            # Store compressed embeddings
            for idx, (doc_id, compressed_emb) in enumerate(zip(ids, compressed_embeddings)):
                embedding_path = os.path.join(data_path, f"{doc_id}.emb")
                with open(embedding_path, 'wb') as f:
                    f.write(compressed_emb)

            # Store in ChromaDB with validation
            cleaned_documents = [str(doc) for doc in documents]  # Ensure all documents are strings
            cleaned_metadatas = [{**meta, 'compressed': True} for meta in metadatas]

            super().add_documents(
                documents=cleaned_documents,
                embeddings=embeddings,
                metadatas=cleaned_metadatas,
                ids=ids
            )

        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise

    def query(self, query_embeddings: List[List[float]], n_results: int,
              include: Optional[List[str]] = None) -> dict:
        """Query with optimized embedding loading."""
        results = super().query(query_embeddings, n_results, include)

        # Load and decompress embeddings only when needed
        if results.get('ids'):
            for doc_id in results['ids'][0]:
                if doc_id not in self.embedding_cache:
                    embedding_path = os.path.join(self.storage_path, f"{doc_id}.emb")
                    if os.path.exists(embedding_path):
                        with open(embedding_path, 'rb') as f:
                            compressed_data = f.read()
                        quantized = decompress_embedding(compressed_data)
                        embedding = self.storage_optimizer.dequantize_embedding(quantized)
                        self.embedding_cache[doc_id] = embedding

        return results

    def clear_cache(self):
        """Clear the embedding cache to free memory."""
        self.embedding_cache.clear()


class RAPTORSystem:
    """Main RAG system orchestrating components."""
    def __init__(
            self,
            embedding_model: HuggingFaceEmbedding,
            vector_store: ChromaDBStore,
            llm: HuggingFaceLLM,
            embedding_model_sentece: SentenceTransformer,
            logger: Optional[logging.Logger] = None
    ):
        self.raptor_tree = RAPTORTree()
        self.prompt_transformer = PromptTransformer(llm)
        self.relevance_scorer = RelevanceScorer()
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm = llm
        self.semantic_analyzer = SemanticAnalyzer(embedding_model_sentece)
        self.logger = logger or self._setup_default_logger()

    def summarize_content(self, document_content: str) -> str:
        """Generates a summary of the document content using generate_summary."""
        return self.llm.generate_summary(document_content)

    @staticmethod
    def _setup_default_logger() -> logging.Logger:
        """Create a default logger with file rotation."""
        logger = logging.getLogger("RAGSystem")
        logger.setLevel(logging.INFO)
        handler = RotatingFileHandler(
            "raptor_system.log",
            maxBytes=1024 * 1024,  # 1MB
            backupCount=5
        )
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def query(self, question: str, n_results: int = 3) -> QueryResult:
        # Step 1: Retrieve relevant history
        relevant_history = self._get_relevant_history(question)

        # Step 2: Transform the original query
        transformed_query = self.prompt_transformer.transform_prompt(
            question, context=relevant_history[0].context if relevant_history else None
        )

        # Step 3: Use hierarchical dynamic routing to retrieve relevant context
        context = self.raptor_tree.hierarchical_query(transformed_query, self.semantic_analyzer, self.relevance_scorer)

        # Step 4: Embed the transformed query
        query_embedding = self.embedding_model.embed([transformed_query])[0]

        # Step 5: Retrieve directly from the vector store for additional context
        results = self.vector_store.query(query_embeddings=[query_embedding.tolist()], n_results=n_results)
        additional_contexts = results.get("documents", [[]])[0]
        refined_context = "\n".join(sorted([context] + additional_contexts[:2],
                                           key=lambda x: -self.relevance_scorer.score_relevance(transformed_query, [x])[
                                               0]))

        # Step 6: Generate response using the LLM
        prompt = f"Given the context:\n{refined_context}\n\nAnswer the question: {transformed_query}"
        response = self.llm.generate_response_with_context(context=refined_context, prompt=prompt, max_new_tokens=250)

        # Step 7: Score confidence
        confidence_score = self.relevance_scorer.score_relevance(transformed_query, [response])[0]

        # Return detailed query result with transformed query, context, response, and confidence
        return QueryResult(
            query=question,
            transformed_query=transformed_query,
            context=refined_context,
            response=response,
            confidence=confidence_score
        )

    def _get_relevant_history(self, query: str) -> List[QueryResult]:
        if not self.prompt_transformer.history:
            return []
        query_embedding = self.embedding_model.embed([query])[0]
        history_embeddings = self.embedding_model.embed([h.query for h in self.prompt_transformer.history])
        similarities = cosine_similarity([query_embedding], history_embeddings)[0]
        return [self.prompt_transformer.history[i] for i in np.where(similarities > 0.8)[0]]

    def index_documents(self, documents: List[Document], batch_size: int = 50) -> None:
        """Optimized indexing method with proper validation and error handling."""
        try:
            documents = remove_duplicate_documents(documents)
            enable_mixed_precision(self.embedding_model.model)

            for i in range(0, len(documents), batch_size):
                batch_documents = documents[i:i + batch_size]
                logger.info(f"Processing batch {i // batch_size + 1}")

                # Create DataFrame with validation
                df = pd.DataFrame({
                    "context": [doc.content for doc in batch_documents],
                    "source_doc": [doc.id for doc in batch_documents]
                })

                # Validate DataFrame
                if df.empty:
                    logger.warning("Empty batch encountered, skipping...")
                    continue

                if df.isnull().any().any():
                    logger.warning("NaN values found in batch, cleaning...")
                    df = df.fillna("")  # Or another appropriate cleaning strategy

                # Create embeddings with validation
                chunked_texts, chunked_doc_ids, embeddings = create_document_embeddings(
                    embedding_model=self.embedding_model,
                    dataframe=df,
                    context_field="context",
                    doc_id_field="source_doc"
                )

                # Validate lengths before adding to vector store
                if not (len(chunked_texts) == len(chunked_doc_ids) == len(embeddings)):
                    raise ValueError(
                        f"Mismatch in lengths: texts={len(chunked_texts)}, "
                        f"ids={len(chunked_doc_ids)}, embeddings={len(embeddings)}"
                    )

                # Create documents with validated data
                chunked_documents = [
                    Document(
                        content=text,
                        metadata={"source": doc_id},
                        id=f"{doc_id}_{idx}"
                    )
                    for idx, (text, doc_id) in enumerate(zip(chunked_texts, chunked_doc_ids))
                ]

                # Add to vector store with validated lengths
                self.vector_store.add_documents(
                    documents=[doc.content for doc in chunked_documents],
                    embeddings=embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings,
                    metadatas=[doc.metadata for doc in chunked_documents],
                    ids=[doc.id for doc in chunked_documents]
                )

                logger.info(f"Successfully indexed batch {i // batch_size + 1}")
                clear_gpu_memory()

        except Exception as e:
            logger.error(f"Failed to index documents: {str(e)}")
            raise


class OptimizedRAPTORSystem(RAPTORSystem):
    """Storage-optimized version of RAPTORSystem."""

    def __init__(self, embedding_model, vector_store, embedding_model_sentece, llm,
                 cache_size: int = 1000, logger: Optional[logging.Logger] = None):
        super().__init__(embedding_model, vector_store, llm, embedding_model_sentece, logger)
        self.cache_size = cache_size
        self._setup_storage_optimization()

    def _setup_storage_optimization(self):
        """Configure storage optimization settings."""
        # Set up periodic cache clearing
        self.query_count = 0
        self.cache_clear_threshold = 100

    def query(self, question: str, n_results: int = 3) -> QueryResult:
        """Query the RAPTOR system, using summaries as context for generating answers."""

        # Retrieve relevant summaries or contexts from the vector store
        query_embedding = self.embedding_model.embed([question])[0]
        results = self.vector_store.query(query_embeddings=[query_embedding.tolist()], n_results=n_results)

        # Use summaries for answer generation if available
        contexts = results.get("documents", [[]])[0]
        refined_context = "\n".join(contexts[:2])  # Use the top 2 summaries as context

        # Generate a response based on the retrieved summaries
        prompt = f"Given the context:\n{refined_context}\n\nAnswer the question: {question}"
        response = self.llm.generate_response_with_context(context=refined_context, prompt=prompt, max_new_tokens=250)

        # Score relevance to compute confidence level
        confidence_score = self.relevance_scorer.score_relevance(question, [response])[0]

        return QueryResult(query=question, transformed_query=question, context=refined_context, response=response,
                           confidence=confidence_score)


class EnhancedRetrievalRAPTOR:
    """Enhanced RAPTOR system with improved retrieval accuracy."""

    def __init__(
            self,
            embedding_model,
            vector_store,
            llm,
            cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
            reranking_top_k: int = 10,
            similarity_threshold: float = 0.85,
            use_hybrid_search: bool = True,
            documents: List[Union[Dict[str, Any], Document]] = None,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm = llm
        self.cross_encoder = CrossEncoder(cross_encoder_name)
        self.reranking_top_k = reranking_top_k
        self.similarity_threshold = similarity_threshold
        self.use_hybrid_search = use_hybrid_search
        self.documents = documents
        self.raptor_tree = RAPTORTree()  # Initialize the hierarchical structure
        self.semantic_analyzer = SemanticAnalyzer(embedding_model)
        self.relevance_scorer = RelevanceScorer()

    def add_document_to_raptor(self, document):
        """Process and add document to the RAPTOR tree with hierarchical structure."""

        # Ensure embedding_model is available
        document_processor = DocumentProcessor(chunk_size=1000, embedding_model=self.embedding_model)

        nodes = document_processor.process_document_to_nodes(document)  # Create nodes from document
        for node in nodes:
            if node.parent_id and not self.semantic_analyzer.verify_semantic_coherence(
                    self.raptor_tree.nodes[node.parent_id], [node]):
                continue  # Skip adding nodes that don't meet coherence criteria
            self.raptor_tree.add_node(node)

    def get_relevant_nodes(self, question: str, nodes: List[RAPTORNode]) -> List[RAPTORNode]:
        """Find and return nodes relevant to the query based on semantic similarity."""
        # Generate the embedding for the question
        query_embedding = self.semantic_analyzer.embedding_model.encode(question)

        relevant_nodes = []

        for node in nodes:
            # Compute semantic similarity between query and each node's embedding
            similarity_score = self.semantic_analyzer.compute_semantic_similarity(query_embedding, node.embedding)

            # If similarity is above a defined threshold, consider it relevant
            if similarity_score >= self.semantic_analyzer.coherence_threshold:
                # Update the node's relevance score (this can be used for sorting)
                node.relevance_score = similarity_score
                relevant_nodes.append(node)

        # Sort relevant nodes by relevance score in descending order
        relevant_nodes.sort(key=lambda x: x.relevance_score, reverse=True)

        return relevant_nodes

    def hierarchical_query(self, question: str, n_results: int = 3) -> QueryResult:
        """Query RAPTOR tree hierarchically, refining context as relevance increases."""
        current_nodes = self.raptor_tree.get_top_nodes()  # Start at the top level
        relevant_contexts = []

        while current_nodes:
            # Filter for relevant nodes based on query similarity
            relevant_nodes = self.get_relevant_nodes(question, current_nodes)
            if not relevant_nodes:
                break
            current_nodes = [child for node in relevant_nodes for child in self.raptor_tree.get_children(node)]
            relevant_contexts.extend([node.content for node in relevant_nodes])

        # Combine relevant contexts and generate response
        refined_context = "\n".join(relevant_contexts[:2])  # Limit for context length
        prompt = f"Given the context:\n{refined_context}\n\nAnswer the question: {question}"
        response = self.llm.generate_response_with_context(context=refined_context, prompt=prompt, max_new_tokens=250)
        confidence_score = self.relevance_scorer.score_relevance(question, [response])[0]

        return QueryResult(query=question, transformed_query=question, context=refined_context, response=response,
                           confidence=confidence_score)

    def _semantic_search(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Perform semantic search using dense embeddings."""
        query_embedding = self.embedding_model.embed([query])[0]
        results = self.vector_store.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )

        # Handle None or empty results
        if results is None:
            logger.warning("No results returned from vector store query.")
            return []

        return self._format_results(results)

    def _hybrid_search(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Combine semantic and keyword-based search."""
        # Semantic search results
        semantic_results = self._semantic_search(query, n_results)

        # BM25 keyword search (implement using your preferred library)
        keyword_results = self._bm25_search(query, n_results)

        # Combine and deduplicate results
        combined_results = self._merge_search_results(
            semantic_results,
            keyword_results,
            weights=[0.7, 0.3]  # Adjust weights based on your needs
        )

        return combined_results[:n_results]

    def _rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank results using cross-encoder."""
        if not results:
            return []

        # Prepare pairs for cross-encoder
        pairs = [[query, result['content']] for result in results]

        # Get cross-encoder scores
        scores = self.cross_encoder.predict(pairs)

        # Sort results by cross-encoder scores
        reranked_results = [
            {**result, 'score': float(score)}
            for result, score in zip(results, scores)
        ]
        reranked_results.sort(key=lambda x: x['score'], reverse=True)

        return reranked_results

    def _filter_by_similarity(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter results based on similarity threshold."""
        return [
            result for result in results
            if result.get('score', 0) >= self.similarity_threshold
        ]

    def _generate_query_variations(self, query: str) -> List[str]:
        """Generate variations of the query for better coverage."""
        # Use LLM to generate query variations
        prompt = f"""Generate 3 alternative ways to ask this question, 
        preserving the core meaning but varying the phrasing:
        Question: {query}

        Format: Return only the questions, one per line."""

        variations = self.llm.generate_response_with_context(
            context="",
            prompt=prompt,
            max_new_tokens=150
        )

        # Clean and parse variations
        variation_list = [q.strip() for q in variations.split('\n') if q.strip()]
        variation_list.append(query)  # Include original query

        return variation_list

    def _aggregate_context(self, contexts: List[str], query: str) -> str:
        """Intelligently aggregate multiple contexts."""
        # Use LLM to analyze and combine contexts
        prompt = f"""Analyze these passages and combine the most relevant information 
        to answer this question: {query}

        Passages:
        {' '.join(contexts)}

        Instructions:
        1. Identify key information relevant to the question
        2. Remove redundant information
        3. Organize information logically
        4. Maintain important details and context

        Combined context:"""

        aggregated = self.llm.generate_response_with_context(
            context="",
            prompt=prompt,
            max_new_tokens=500
        )

        return aggregated

    def enhanced_query(self, question: str, n_results: int = 3) -> Dict[str, Any]:
        """Enhanced query processing with multiple improvements."""
        # Generate query variations
        query_variations = self._generate_query_variations(question)

        all_results = []
        for query in query_variations:
            # Use hybrid search if enabled, otherwise semantic search
            if self.use_hybrid_search:
                results = self._hybrid_search(query, self.reranking_top_k)
            else:
                results = self._semantic_search(query, self.reranking_top_k)

            # Rerank results
            reranked_results = self._rerank_results(query, results)

            # Filter by similarity
            filtered_results = self._filter_by_similarity(reranked_results)

            all_results.extend(filtered_results)

        # Deduplicate and get top results
        unique_results = self._deduplicate_results(all_results)
        top_results = unique_results[:n_results]

        # Aggregate context
        contexts = [result['content'] for result in top_results]
        aggregated_context = self._aggregate_context(contexts, question)

        # Generate final response
        final_response = self._generate_response(question, aggregated_context)

        return {
            'question': question,
            'response': final_response,
            'context': aggregated_context,
            'confidence': safe_mean([result['score'] for result in top_results])
        }

    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results while preserving the highest scoring ones."""
        seen_content = set()
        unique_results = []

        for result in sorted(results, key=lambda x: x.get('score', 0), reverse=True):
            # Create a normalized version of the content for comparison
            normalized_content = ' '.join(result['content'].lower().split())

            if normalized_content not in seen_content:
                seen_content.add(normalized_content)
                unique_results.append(result)

        return unique_results

    def _generate_response(self, question: str, context: str) -> str:
        """Generate final response with enhanced prompt."""
        prompt = f"""Based on the following context, provide a comprehensive and accurate answer to the question.
        If the context doesn't contain enough information, acknowledge the limitations.

        Context: {context}

        Question: {question}

        Instructions:
        1. Answer directly and precisely
        2. Include relevant supporting details from the context
        3. Maintain accuracy and avoid speculation
        4. Acknowledge any information gaps

        Answer:"""

        response = self.llm.generate_response_with_context(
            context=context,
            prompt=prompt,
            max_new_tokens=300
        )

        return response

    def _bm25_search(self, query: str, n_results: int) -> List[Dict[str, Any]]:
        """Implement BM25 keyword search using rank_bm25 library.

        Args:
            query (str): The search query string
            n_results (int): Number of results to return

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing search results with scores
        """

        # Tokenize the query
        query_tokens = query.lower().split()

        # If we haven't preprocessed the corpus yet, do it now
        if not hasattr(self, '_bm25'):
            # Tokenize all documents
            tokenized_corpus = [doc['content'].lower().split()
                                for doc in self.documents]
            # Initialize BM25 with our corpus
            self._bm25 = BM25Okapi(tokenized_corpus)
            # Store original documents for later retrieval
            self._doc_mapping = self.documents

        # Get BM25 scores for all documents
        doc_scores = self._bm25.get_scores(query_tokens)

        # Get indices of top n_results documents, sorted by score
        top_indices = sorted(range(len(doc_scores)),
                             key=lambda i: doc_scores[i],
                             reverse=True)[:n_results]

        # Prepare results
        results = []
        for idx in top_indices:
            if doc_scores[idx] > 0:  # Only include relevant results
                results.append({
                    'document': self._doc_mapping[idx],
                    'score': float(doc_scores[idx])  # Convert numpy float to Python float
                })

        return results

    def _merge_search_results(
            self,
            semantic_results: List[Dict[str, Any]],
            keyword_results: List[Dict[str, Any]],
            weights: List[float]
    ) -> List[Dict[str, Any]]:
        """Merge and score results from different search methods."""
        merged_dict = {}

        # Process semantic results
        for result in semantic_results:
            content = result.get('content')  # Safely get 'content', or return None if missing
            if content:
                merged_dict[content] = {
                    'score': result.get('score', 0) * weights[0],
                    'metadata': result.get('metadata', {})
                }

        # Process keyword results, only if they are not None
        if keyword_results:
            for result in keyword_results:
                content = result.get('content')  # Safely get 'content'
                if content:
                    if content in merged_dict:
                        merged_dict[content]['score'] += result.get('score', 0) * weights[1]
                    else:
                        merged_dict[content] = {
                            'score': result.get('score', 0) * weights[1],
                            'metadata': result.get('metadata', {})
                        }

        # Convert back to list format
        merged_results = [
            {
                'content': content,
                'score': data['score'],
                'metadata': data['metadata']
            }
            for content, data in merged_dict.items()
        ]

        # Sort by combined score
        merged_results.sort(key=lambda x: x['score'], reverse=True)

        return merged_results

    def _format_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format vector store results into standard format."""
        formatted_results = []

        documents = results.get('documents', [[]])[0] if results.get('documents') else []
        metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else []
        distances = results.get('distances', [[]])[0] if results.get('distances') else []

        for doc, meta, dist in zip(documents, metadatas, distances):
            formatted_results.append({
                'content': doc,
                'metadata': meta,
                'score': 1 - dist  # Convert distance to similarity score
            })

        return formatted_results


def clear_gpu_memory():
    """Clears GPU memory and forces garbage collection."""
    torch.cuda.empty_cache()
    gc.collect()
    log_cuda_memory_usage("Cleared GPU memory")


def log_cuda_memory_usage(message=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        logger.info(f"{message} - CUDA memory allocated: {allocated:.2f} MB, reserved: {reserved:.2f} MB")
    else:
        logger.info(f"{message} - CUDA not available")


def enable_mixed_precision(model):
    """Move model to half precision if the model and GPU support it."""
    if torch.cuda.is_available() and hasattr(model, "half"):
        model.half()
        logger.info("Enabled mixed-precision for the model.")


def remove_duplicate_documents(documents):
    """Remove documents with duplicate IDs to avoid indexing redundancy."""
    unique_documents = {}
    for doc in documents:
        if doc.id not in unique_documents:
            unique_documents[doc.id] = doc
    return list(unique_documents.values())


def find_parent_id(index, level, nodes):
    """Find parent node id for the current chunk based on its level."""
    if level == 1:
        return None  # Top level has no parent
    for node in reversed(nodes):
        if node.level == level - 1:
            return node.node_id
    return None


def determine_level(chunk):
    """Determine the hierarchical level of a document chunk based on structure."""
    # Placeholder logic - adapt based on document format
    if "Chapter" in chunk:
        return 1
    elif "Section" in chunk:
        return 2
    return 3


def default_summarize_content(text: str) -> str:
    """A placeholder summarization function."""
    return text[:200]  # A simple fallback that truncates content as a 'summary'


class DocumentProcessor:
    def __init__(
            self,
            chunk_size: int = 1000,
            chunk_overlap: int = 115,
            raptor_system: Optional[RAPTORSystem] = None,
            embedding_model: Optional[HuggingFaceEmbedding] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # Type hint to indicate HuggingFaceEmbedding is expected
        self.embedding_model: HuggingFaceEmbedding = embedding_model
        if self.embedding_model is None:
            raise ValueError("An embedding model of type HuggingFaceEmbedding must be provided.")

        # Summarize content function based on provided RAPTORSystem or default
        self.summarize_content = raptor_system.summarize_content if raptor_system else default_summarize_content

    def process_document_to_nodes(self, document):
        nodes = []
        chunks = self.text_splitter.split_text(document.content)
        for i, chunk in enumerate(chunks):
            level = determine_level(chunk)
            parent_id = find_parent_id(i, level, nodes)

            # Generate embedding using the `embed` method
            embedding = self.embedding_model.embed([chunk])[
                0]  # `embed` returns a batch, so use [0] to get the first embedding

            node = RAPTORNode(
                content=chunk,
                summary=self.summarize_content(chunk),
                embedding=embedding,
                level=level,
                node_id=f"{document.id}_{i}",
                parent_id=parent_id
            )
            nodes.append(node)
        return nodes

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


def calculate_storage_requirements(num_documents: int, embedding_dim: int,
                                   avg_document_size: int) -> float:
    """Calculate approximate storage requirements in GB."""
    # Assuming compressed embeddings (8-bit quantization + compression)
    embedding_size = embedding_dim / 4  # Compressed size estimate
    total_embedding_storage = (embedding_size * num_documents) / (1024 ** 3)

    # Document storage (assuming text compression)
    document_storage = (avg_document_size * num_documents * 0.3) / (1024 ** 3)

    # Index overhead (approximately 10%)
    index_overhead = (total_embedding_storage + document_storage) * 0.1

    return total_embedding_storage + document_storage + index_overhead


def create_document_embeddings(
        embedding_model,
        dataframe,
        context_field="context",
        doc_id_field="source_doc",
        batch_size=16,
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


class UnifiedRAPTORSystem:
    """Unified RAPTOR System combining base, enhanced retrieval, and optimized functionality."""
    def __init__(
            self,
            embedding_model: HuggingFaceEmbedding,
            vector_store: ChromaDBStore,
            llm: HuggingFaceLLM,
            embedding_model_sentece: SentenceTransformer,
            cache_size: int = 1000,
            cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
            reranking_top_k: int = 10,
            similarity_threshold: float = 0.65,
            use_hybrid_search: bool = True,
            logger: Optional[logging.Logger] = None,
    ):
        # Core components
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm = llm
        self.semantic_analyzer = SemanticAnalyzer(embedding_model_sentece)
        self.relevance_scorer = RelevanceScorer(cross_encoder_name)
        self.prompt_transformer = PromptTransformer(llm)

        # RAPTOR-specific components
        self.raptor_tree = RAPTORTree()

        # Optimization parameters
        self.cache_size = cache_size
        self.reranking_top_k = reranking_top_k
        self.similarity_threshold = similarity_threshold
        self.use_hybrid_search = use_hybrid_search

        # Logger
        self.logger = logger or self._setup_default_logger()

        # Query cache for optimized retrieval
        self.query_cache = {}
        self.cache_clear_threshold = 100
        self.query_count = 0

    def initialize_bm25(self, documents: List[Dict[str, Any]]):
        """Initialize the BM25 model with tokenized documents."""
        self._doc_mapping = documents  # Save the original document mappings
        tokenized_corpus = [doc['content'].lower().split() for doc in documents]
        self._bm25 = BM25Okapi(tokenized_corpus)

    @staticmethod
    def _setup_default_logger() -> logging.Logger:
        """Create a default logger with file rotation."""
        logger = logging.getLogger("UnifiedRAPTORSystem")
        logger.setLevel(logging.INFO)
        handler = RotatingFileHandler(
            "raptor_system.log",
            maxBytes=1024 * 1024,  # 1MB
            backupCount=5
        )
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def index_documents(self, documents: List[Document], batch_size: int = 50) -> None:
        """Index documents into the vector store."""
        try:
            documents = remove_duplicate_documents(documents)
            enable_mixed_precision(self.embedding_model.model)

            for i in range(0, len(documents), batch_size):
                batch_documents = documents[i:i + batch_size]
                self.logger.info(f"Processing batch {i // batch_size + 1}")

                chunked_texts, chunked_doc_ids, embeddings = create_document_embeddings(
                    embedding_model=self.embedding_model,
                    dataframe=pd.DataFrame({
                        "context": [doc.content for doc in batch_documents],
                        "source_doc": [doc.id for doc in batch_documents]
                    }),
                    context_field="context",
                    doc_id_field="source_doc"
                )

                self.vector_store.add_documents(
                    documents=chunked_texts,
                    embeddings=embeddings.tolist(),
                    metadatas=[{"source": doc_id} for doc_id in chunked_doc_ids],
                    ids=[f"{doc_id}_{idx}" for idx, doc_id in enumerate(chunked_doc_ids)]
                )
                self.logger.info(f"Successfully indexed batch {i // batch_size + 1}")
                clear_gpu_memory()
        except Exception as e:
            self.logger.error(f"Failed to index documents: {str(e)}")
            raise

    def query(self, question: str, n_results: int = 3) -> QueryResult:
        """Handle query with hierarchical, hybrid, and optimized retrieval."""
        # Step 1: Retrieve relevant query history
        relevant_history = self._get_relevant_history(question)

        # Step 2: Transform the query
        transformed_query = self.prompt_transformer.transform_prompt(
            question, context=relevant_history[0].context if relevant_history else None
        )

        # Step 3: Hierarchical retrieval
        context = self.raptor_tree.hierarchical_query(
            question=transformed_query,
            analyzer=self.semantic_analyzer,
            scorer=self.relevance_scorer
        )

        # Step 4: Embed the transformed query
        query_embedding = self.embedding_model.embed([transformed_query])[0]

        # Step 5: Hybrid or semantic search in the vector store
        search_results = self._hybrid_search(transformed_query) if self.use_hybrid_search else self._semantic_search(
            transformed_query)
        additional_contexts = [res['content'] for res in search_results[:n_results]]

        # Step 6: Aggregate context
        combined_context = self._aggregate_context([context] + additional_contexts, transformed_query)

        # Step 7: Generate response using the LLM
        response = self._generate_response(transformed_query, combined_context)

        # Step 8: Calculate confidence score
        confidence_score = safe_mean([self.relevance_scorer.score_relevance(transformed_query, [res])[0]
                                      for res in additional_contexts])

        return QueryResult(
            query=question,
            transformed_query=transformed_query,
            context=combined_context,
            response=response,
            confidence=confidence_score
        )

    def _hybrid_search(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Perform hybrid search combining semantic and keyword-based retrieval."""
        semantic_results = self._semantic_search(query, n_results)
        keyword_results = self._bm25_search(query, n_results)
        return self._merge_search_results(semantic_results, keyword_results)

    def _semantic_search(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Perform semantic search using vector store."""
        query_embedding = self.embedding_model.embed([query])[0]
        results = self.vector_store.query([query_embedding.tolist()], n_results=n_results)

        # Validate the structure of results
        if not results or not results.get("documents"):
            logger.warning("Semantic search returned no results.")
            return []

        formatted_results = []
        for doc, meta, dist in zip(
                results.get("documents", [[]])[0],
                results.get("metadatas", [[]])[0],
                results.get("distances", [[]])[0]
        ):
            formatted_results.append({
                'content': doc,
                'metadata': meta,
                'score': 1 - dist  # Convert distance to similarity score
            })
        return formatted_results

    def _aggregate_context(self, contexts: List[str], query: str) -> str:
        """Combine multiple contexts into a coherent response."""
        prompt = f"""Combine the following contexts to answer the question:\n\n{query}\n\nContexts:\n{contexts}"""
        return self.llm.generate_response_with_context(prompt=prompt, context="", max_new_tokens=500)

    def _generate_response(self, query: str, context: str) -> str:
        """Generate the final response based on the aggregated context."""
        prompt = f"""Answer the question based on the context below:\n\nContext: {context}\n\nQuestion: {query}"""
        return self.llm.generate_response_with_context(prompt=prompt, context=context, max_new_tokens=300)

    def _get_relevant_history(self, query: str) -> List[QueryResult]:
        """Retrieve relevant past queries for context."""
        if not self.prompt_transformer.history:
            return []
        query_embedding = self.embedding_model.embed([query])[0]
        history_embeddings = self.embedding_model.embed([h.query for h in self.prompt_transformer.history])
        similarities = cosine_similarity([query_embedding], history_embeddings)[0]
        return [self.prompt_transformer.history[i] for i in np.where(similarities > 0.8)[0]]

    def _bm25_search(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Perform BM25 keyword search."""
        query_tokens = query.lower().split()

        # Ensure BM25 is initialized
        if not hasattr(self, '_bm25'):
            raise ValueError("BM25 model not initialized. Call `initialize_bm25` before using BM25 search.")

        doc_scores = self._bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:n_results]

        results = []
        for idx in top_indices:
            results.append({
                'content': self._doc_mapping[idx]['content'],
                'metadata': self._doc_mapping[idx].get('metadata', {}),
                'score': float(doc_scores[idx])
            })
        return results

    def _merge_search_results(
            self,
            semantic_results: List[Dict[str, Any]],
            keyword_results: List[Dict[str, Any]],
            semantic_weight: float = 0.7,
            keyword_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Merge and score results from semantic and keyword-based searches."""
        # Ensure all results are dictionaries
        if not all(isinstance(res, dict) for res in semantic_results + keyword_results):
            raise TypeError(
                "All search results must be dictionaries with keys like 'content', 'metadata', and 'score'.")

        merged_results = {}

        # Add semantic results with weighted scores
        for result in semantic_results:
            content = result['content']
            score = result.get('score', 0) * semantic_weight
            if content in merged_results:
                merged_results[content]['score'] += score
            else:
                merged_results[content] = {
                    'content': content,
                    'metadata': result.get('metadata', {}),
                    'score': score
                }

        # Add keyword results with weighted scores
        for result in keyword_results:
            content = result['content']
            score = result.get('score', 0) * keyword_weight
            if content in merged_results:
                merged_results[content]['score'] += score
            else:
                merged_results[content] = {
                    'content': content,
                    'metadata': result.get('metadata', {}),
                    'score': score
                }

        # Convert merged results to a list and sort by score
        merged_results_list = list(merged_results.values())
        merged_results_list.sort(key=lambda x: x['score'], reverse=True)

        return merged_results_list


def main():
    # Configuration
    available_space = 22.09  # GB
    embedding_dim = 1024  # Adjust based on your model
    avg_document_size = 1000  # bytes

    # Calculate maximum documents based on available space
    max_documents = int((available_space * 0.8 * (1024 ** 3)) /
                        (embedding_dim / 4 + avg_document_size * 0.3 * 1.1))

    # Initialize components
    embedding_model = HuggingFaceEmbedding(
        model_name="dunzhang/stella_en_1.5B_v5",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    vector_store = OptimizedChromaDBStore(
        collection_name="optimized_data_bank",
        storage_path="./optimized_storage/",
        max_documents=max_documents
    )

    llm = HuggingFaceLLM(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float16
    )

    semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Initialize the Unified RAPTOR System
    raptor_system = UnifiedRAPTORSystem(
        embedding_model=embedding_model,
        vector_store=vector_store,
        llm=llm,
        embedding_model_sentece=semantic_model,
        cache_size=1000,
        cross_encoder_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
        reranking_top_k=10,
        similarity_threshold=0.65,
        use_hybrid_search=True
    )

    # Load and process documents
    doc_processor = DocumentProcessor(chunk_size=1000, embedding_model=embedding_model)
    documents = doc_processor.load_and_split_documents(Path(DEFAULT_DATA_PATH))

    # Convert Document objects to dictionaries
    documents_as_dicts = [{"content": doc.content, "metadata": doc.metadata, "id": doc.id} for doc in documents]

    # Initialize BM25 for keyword search
    raptor_system.initialize_bm25(documents_as_dicts)

    # Adjust document count if storage is limited
    estimated_storage = calculate_storage_requirements(len(documents), embedding_dim, avg_document_size)
    if estimated_storage > available_space:
        logger.warning(
            f"Estimated storage requirement ({estimated_storage:.2f} GB) exceeds available space ({available_space} GB)"
        )
        reduction_factor = available_space / estimated_storage
        documents = documents[:int(len(documents) * reduction_factor)]

    # Index documents into the system
    raptor_system.index_documents(documents)

    # Define example questions
    questions = [
        "Is Hirschsprung disease a Mendelian or a multifactorial disorder?",
        "List signaling molecules (ligands) that interact with the receptor EGFR?",
        "Are long non-coding RNAs spliced?",
        "Is RANKL secreted from the cells?",
        "Which miRNAs could be used as potential biomarkers for epithelial ovarian cancer?",
        "Which acetylcholinesterase inhibitors are used for treatment of myasthenia gravis?",
        "Has Denosumab (Prolia) been approved by FDA?",
        "Which are the different isoforms of the mammalian Notch receptor?",
        "Orteronel was developed for treatment of which cancer?",
        "Is the monoclonal antibody Trastuzumab (Herceptin) of potential use in the treatment of prostate cancer?",
        "Which are the Yamanaka factors?",
        "Where is the protein Pannexin1 located?",
        "Which currently known mitochondrial diseases have been attributed to POLG mutations?"
    ]

    # Query the Unified RAPTOR System
    results = {}
    for question in questions:
        result = raptor_system.query(question, n_results=3)
        results[question] = {
            "response": result.response,
            "context": result.context,
            "confidence": result.confidence
        }
        print(f"\n❀❀ Question ❀❀: {question}")
        print(f"❀❀ Answer ❀❀: {result.response}")

    # Save results to JSON
    with open("unified_raptor_results.json", "w") as file:
        json.dump(results, file, indent=4)
    print("Results saved to unified_raptor_results.json")


if __name__ == "__main__":
    main()

