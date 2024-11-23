import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from dataclasses import dataclass

from my_rag.evaluations.evaluator import get_dataset_loader
from my_rag.components.pipeline.document_processor import DocumentProcessor
from my_rag.components.embeddings.aws_embedding import AWSBedrockEmbedding
from my_rag.components.pipeline.embedder import DocumentEmbedder, QueryEmbedder
from my_rag.components.pipeline.retriever import Retriever
from my_rag.components.pipeline.generator import Generator
from my_rag.components.pipeline.rag_pipeline import RAGPipeline
from my_rag.components.utils import alphanumeric_string
from my_rag.components.embeddings.huggingface_embedding import HuggingFaceEmbedding
from my_rag.components.llms.huggingface_llm import HuggingFaceLLM
from my_rag.components.llms.aws_llm import AWSBedrockLLM
from my_rag.components.vectorstores.chroma_store import (
    ChromaVectorStore,
    CollectionMode,
)
import configparser

logger = logging.getLogger(__name__)


@dataclass
class RAGEvaluationConfig:
    """Configuration for RAG evaluation"""

    dataset_configs: List[Dict[str, Any]]
    embedding_model_configs: List[Dict[str, Any]]
    llm_model_configs: List[Dict[str, Any]]
    max_k: int = 5
    chunk_size: int = 2000
    chunk_overlap: int = 250
    output_dir: str = "results"


class RAGEvaluator:
    """Evaluates complete RAG pipelines including generation"""

    def __init__(self, config: RAGEvaluationConfig):
        self.config = config

    def _create_pipeline(
        self, embedding_config: Dict[str, Any], llm_config: Dict[str, Any]
    ):
        """Creates RAG pipeline with both retrieval and generation components"""

        if embedding_config.get("model_kwargs", {}).get("aws"):
            config_file = embedding_config.get("model_kwargs", {}).get("aws_creds_file")
            config_name = embedding_config.get("model_kwargs", {}).get(
                "aws_config_name"
            )
            config = configparser.ConfigParser()
            config.read(config_file)
            aws_access_key = config[config_name]["aws_access_key_id"]
            aws_secret_key = config[config_name]["aws_secret_access_key"]
            aws_session_token = config[config_name]["aws_session_token"]
            region = config[config_name]["region"]
            embedding_model = AWSBedrockEmbedding(
                model_id=embedding_config["name"],
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                aws_session_token=aws_session_token,
                region_name=region,
            )
        else:
            embedding_model = HuggingFaceEmbedding(
                model_name=embedding_config["name"],
                **embedding_config.get("model_kwargs", {}),
            )

        if llm_config.get("model_kwargs", {}).get("aws"):
            config_file = llm_config.get("model_kwargs", {}).get("aws_creds_file")
            config_name = llm_config.get("model_kwargs", {}).get("aws_config_name")
            config = configparser.ConfigParser()
            config.read(config_file)
            aws_access_key = config[config_name]["aws_access_key_id"]
            aws_secret_key = config[config_name]["aws_secret_access_key"]
            aws_session_token = config[config_name]["aws_session_token"]
            region = config[config_name]["region"]
            llm_model = AWSBedrockLLM(
                model_id=llm_config["name"],
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                aws_session_token=aws_session_token,
                region_name=region,
            )
        else:
            llm_model = HuggingFaceLLM(
                model_name=llm_config["name"], **llm_config.get("model_kwargs", {})
            )

        # Initialize vector store
        vector_store = ChromaVectorStore(
            collection_name=alphanumeric_string(f"eval_{embedding_config['name']}"),
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
                    batch_size=embedding_config.get("batch_size", 32),
                    instruction=embedding_config.get("instruction"),
                ),
                QueryEmbedder(
                    embedding_model=embedding_model,
                    batch_size=embedding_config.get("batch_size", 32),
                    instruction=embedding_config.get("query_instruction"),
                ),
                Retriever(vector_store=vector_store, k=self.config.max_k),
                Generator.from_config(
                    llm=llm_model, config=llm_config["generator_config"]
                ),
            ]
        )

    def evaluate_models(
        self,
        embedding_config: Dict[str, Any],
        llm_config: Dict[str, Any],
        dataset: Dict[str, Any],
    ) -> pd.DataFrame:
        """Evaluates a specific combination of embedding model and LLM"""

        logger.info(
            f"Evaluating embedding model: {embedding_config['name']} with LLM: {llm_config['name']}"
        )

        pipeline = self._create_pipeline(embedding_config, llm_config)

        # Run pipeline
        pipeline_data = pipeline.run(
            documents=dataset["documents"],
            document_ids=dataset["document_ids"],
            queries=dataset["queries"],
        )

        # Prepare results
        results = []
        for i, (query, answer) in enumerate(
            zip(dataset["queries"], pipeline_data.generated_responses)
        ):
            result = {
                "question": query,
                "ideal_answer": dataset.get("ideal_answers")[i],
                "llm_generated_answer": answer,
                "document_id": dataset["actual_doc_ids"][i],
                "retrieved_document_ids": [
                    m["doc_id"] for m in pipeline_data.retrieved_metadata[i]
                ],
                "contexts": pipeline_data.retrieved_documents[i],
            }
            results.append(result)

        return pd.DataFrame(results)

    def evaluate_all(self):
        """Evaluates all model combinations across all datasets"""

        for dataset_config in self.config.dataset_configs:
            # Load dataset
            dataset = get_dataset_loader(dataset_config["type"]).load(dataset_config)

            for embedding_config in self.config.embedding_model_configs:
                for llm_config in self.config.llm_model_configs:
                    # Evaluate model combination
                    results_df = self.evaluate_models(
                        embedding_config=embedding_config,
                        llm_config=llm_config,
                        dataset=dataset,
                    )

                    # Save results
                    output_path = (
                        Path(self.config.output_dir)
                        / f'rag_evaluations_{embedding_config["name"].replace("/", "_")}_{llm_config["name"].replace("/", "_")}.xlsx'
                    )
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    results_df.to_excel(output_path, index=False)
                    logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG pipelines")
    parser.add_argument("--config", required=True, help="Path to config file")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create evaluation config
    eval_config = RAGEvaluationConfig(
        dataset_configs=config["datasets"],
        embedding_model_configs=config["embedding_models"],
        llm_model_configs=config["llm_models"],
        max_k=config.get("max_k", 5),
        chunk_size=config.get("chunk_size", 2000),
        chunk_overlap=config.get("chunk_overlap", 250),
        output_dir=config.get("output_dir", "results"),
    )

    # Run evaluation
    evaluator = RAGEvaluator(eval_config)
    evaluator.evaluate_all()

    logging.info("RAG evaluation completed successfully")


if __name__ == "__main__":
    main()
