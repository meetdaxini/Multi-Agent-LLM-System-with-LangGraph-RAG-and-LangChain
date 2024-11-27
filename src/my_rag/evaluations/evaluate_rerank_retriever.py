import yaml
import argparse
from my_rag.evaluations.evaluator import RerankingRetrieverEvaluator, EvaluationConfig
from my_rag.evaluations.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Evaluate retriever models")
    parser.add_argument("--config", required=True, help="Path to config file")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Setup main logger
    logger = setup_logger("main_evaluation")
    logger.info("Starting evaluation process")

    # Create evaluation config
    eval_config = EvaluationConfig(
        dataset_configs=config["datasets"],
        model_configs=config["models"],
        max_k=config.get("max_k", 5),
        rereank_max_k=config.get("rereank_max_k", 5),
        chunk_size=config.get("chunk_size", 2000),
        chunk_overlap=config.get("chunk_overlap", 250),
        output_path=config.get("output_path", "retriever_evaluation_results.xlsx"),
    )

    # Run evaluation
    evaluator = RerankingRetrieverEvaluator(eval_config)
    evaluator.evaluate_all()

    logger.info("Evaluation completed successfully")


if __name__ == "__main__":
    main()
