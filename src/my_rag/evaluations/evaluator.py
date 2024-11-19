from typing import List, Dict, Any
from dataclasses import dataclass
import pandas as pd
import logging
from my_rag.components.pipeline.document_processor import DocumentProcessor
from my_rag.components.pipeline.embedder import DocumentEmbedder, QueryEmbedder
from my_rag.components.pipeline.retriever import Retriever
from my_rag.components.pipeline.rag_pipeline import RAGPipeline
from my_rag.components.utils import alphanumeric_string
from .metrics import MetricsCalculator
import os
from my_rag.components.embeddings.huggingface_embedding import HuggingFaceEmbedding
from my_rag.components.vectorstores.chroma_store import (
    ChromaVectorStore,
    CollectionMode,
)
import pandas as pd
from my_rag.components.pdf_loader import PDFLoader
from typing import Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


# Dataset loaders
class ParquetDatasetLoader:
    def load(self, config: Dict[str, Any]) -> Dict[str, Any]:
        import pandas as pd

        df = pd.read_parquet(config["path"])
        return {
            "documents": df[config["context_field"]].tolist(),
            "document_ids": df[config["doc_id_field"]].tolist(),
            "queries": df[config["question_field"]].tolist(),
            "actual_doc_ids": df[config["doc_id_field"]].tolist(),
        }


class CSVPDFDatasetLoader:
    def load(self, config: Dict[str, Any]) -> Dict[str, Any]:
        pdf_loader = PDFLoader()
        df = pd.read_csv(config["path"])
        df = df.drop_duplicates(subset="Question ID", keep="first")

        # Load PDFs
        documents = []
        for pdf_file in df[config["doc_id_field"]]:
            pdf_path = Path(config["pdf_dir"]) / pdf_file
            try:
                content = pdf_loader.load_single_pdf(str(pdf_path))
                documents.append(content)
            except Exception as e:
                logging.error(f"Error loading PDF {pdf_file}: {e}")
                documents.append("")

        return {
            "documents": documents,
            "document_ids": df[config["doc_id_field"]].tolist(),
            "queries": df[config["question_field"]].tolist(),
            "actual_doc_ids": df[config["doc_id_field"]].tolist(),
        }


def get_dataset_loader(dataset_type: str):
    loaders = {"parquet": ParquetDatasetLoader(), "csv_pdf": CSVPDFDatasetLoader()}
    return loaders[dataset_type]


@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""

    dataset_configs: List[Dict[str, Any]]
    model_configs: List[Dict[str, Any]]
    max_k: int = 5
    chunk_size: int = 2000
    chunk_overlap: int = 250
    output_path: str = "retriever_evaluation_results.xlsx"


class RetrieverEvaluator:
    """Evaluates retriever models using the RAG pipeline"""

    def __init__(
        self,
        config: EvaluationConfig,
    ):
        self.config = config
        self.metrics_calculator = MetricsCalculator()

    def _create_pipeline(self, model_config: Dict[str, Any]):
        """Creates pipeline for a specific model configuration"""
        embedding_model = HuggingFaceEmbedding(
            model_name=model_config["name"], **model_config.get("model_kwargs", {})
        )

        vector_store = ChromaVectorStore(
            collection_name=alphanumeric_string(f"eval_{model_config['name']}"),
            mode=CollectionMode.DROP_IF_EXISTS,
        )
        return RAGPipeline(
            [
                DocumentProcessor(
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap,
                ),
                DocumentEmbedder(
                    embedding_model=embedding_model,
                    batch_size=model_config["batch_size"],
                    instruction=model_config.get("instruction"),
                ),
                QueryEmbedder(
                    embedding_model=embedding_model,
                    batch_size=model_config["batch_size"],
                    instruction=model_config.get("query_instruction"),
                ),
                Retriever(vector_store=vector_store, k=self.config.max_k),
            ]
        )

    def evaluate_model(
        self,
        model_config: Dict[str, Any],
        dataset_config: Dict[str, Any],
        documents: List[str],
        document_ids: List[str],
        queries: List[str],
        actual_doc_ids: List[str],
    ) -> Dict[str, Any]:
        """Evaluates a single model"""
        logger.info(f"Evaluating model: {model_config['name']}")

        pipeline = self._create_pipeline(model_config)

        # Run pipeline
        pipeline_data = pipeline.run(
            documents=documents, document_ids=document_ids, queries=queries
        )

        # Extract retrieved document IDs
        retrieved_doc_ids = []
        for batch in pipeline_data.retrieved_metadata:
            batch_ids = []
            for metadata in batch:
                batch_ids.append(metadata["doc_id"])
            retrieved_doc_ids.append(batch_ids)

        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(
            retrieved_doc_ids=retrieved_doc_ids,
            actual_doc_ids=actual_doc_ids,
            max_k=self.config.max_k,
        )

        result = {
            "model": model_config["name"],
            "dataset": dataset_config["name"],
            "settings": model_config,
        }
        result.update(metrics.to_dict())

        return result

    def evaluate_all(self) -> pd.DataFrame:
        """Evaluates all models and returns results DataFrame"""
        results = []
        for dataset_config in self.config.dataset_configs:
            # Load dataset
            dataset = get_dataset_loader(dataset_config["type"]).load(dataset_config)
            documents = dataset["documents"]
            document_ids = dataset["document_ids"]
            queries = dataset["queries"]
            actual_doc_ids = dataset["actual_doc_ids"]

            for model_config in self.config.model_configs:
                result = self.evaluate_model(
                    model_config=model_config,
                    dataset_config=dataset_config,
                    documents=documents,
                    document_ids=document_ids,
                    queries=queries,
                    actual_doc_ids=actual_doc_ids,
                )
                results.append(result)

        results_df = pd.DataFrame(results)
        self._save_results(results_df)

        return results_df

    def _save_results(self, df: pd.DataFrame):
        """Saves evaluation results"""
        os.makedirs(os.path.dirname(self.config.output_path), exist_ok=True)
        df.to_excel(self.config.output_path, index=False)
        logger.info(f"Results saved to {self.config.output_path}")
