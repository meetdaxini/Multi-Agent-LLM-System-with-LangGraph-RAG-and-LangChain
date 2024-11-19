import yaml
import logging
import argparse
from my_rag.evaluations.evaluator import RetrieverEvaluator, EvaluationConfig


def main():
    parser = argparse.ArgumentParser(description="Evaluate retriever models")
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
    eval_config = EvaluationConfig(
        dataset_configs=config["datasets"],
        model_configs=config["models"],
        max_k=config.get("max_k", 5),
        chunk_size=config.get("chunk_size", 2000),
        chunk_overlap=config.get("chunk_overlap", 250),
        output_path=config.get("output_path", "retriever_evaluation_results.xlsx"),
    )

    # Run evaluation
    evaluator = RetrieverEvaluator(eval_config)
    evaluator.evaluate_all()

    logging.info("Evaluation completed successfully")


if __name__ == "__main__":
    main()
