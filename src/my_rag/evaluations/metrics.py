from dataclasses import dataclass
from typing import List, Dict
from .logger import setup_logger

# import numpy as np


@dataclass
class RetrievalMetrics:
    """Class to hold retrieval evaluation metrics"""

    accuracy_at_k: Dict[int, float]

    def to_dict(self) -> Dict[str, float]:
        metrics = {}
        for k, acc in self.accuracy_at_k.items():
            metrics[f"Accuracy@{k}"] = acc
        return metrics


class MetricsCalculator:
    """Calculator for retrieval metrics"""

    @staticmethod
    def calculate_metrics(
        retrieved_doc_ids: List[List[str]],
        actual_doc_ids: List[List[str]],
        max_k: int,
        model_name: str,
    ) -> RetrievalMetrics:
        logger = setup_logger(model_name)
        accuracy_at_k = {}

        for k in range(1, max_k + 1):
            logger.info(f"\nEvaluating for k={k}")
            correct_at_k = 0

            for query_idx, (retrieved_docs, actual_docs) in enumerate(
                zip(retrieved_doc_ids, actual_doc_ids)
            ):
                logger.info(f"\nQuery {query_idx + 1}:")
                logger.info(f"Actual documents: {actual_docs}")

                # Get top k retrieved docs
                top_k_docs = retrieved_docs[:k]
                logger.info(f"Retrieved top-{k} documents: {top_k_docs}")

                if len(actual_docs) == 1 or k == 1:
                    matched = any(doc in actual_docs for doc in top_k_docs)
                    if matched:
                        correct_at_k += 1
                    logger.info(f"Single document case - Match found: {matched}")
                else:
                    total_docs = len(set(actual_docs))
                    matches = set(top_k_docs).intersection(set(actual_docs))
                    if k <= total_docs:
                        total = k
                    else:  # k > total_docs
                        total = total_docs

                    score = len(matches) / total
                    correct_at_k += score
                    logger.info(f"Multiple documents case:")
                    logger.info(f"Total documents to match: {total}")
                    logger.info(f"Matches found {len(matches)}: {matches}")
                    logger.info(f"Score: {score:.2f}")

            accuracy_at_k[k] = correct_at_k / len(actual_doc_ids)
            logger.info(f"\nOverall Accuracy@{k}: {accuracy_at_k[k]:.4f}")
        # # Calculate MRR
        # for retrieved_docs, actual_doc in zip(retrieved_doc_ids, actual_doc_ids):
        #     try:
        #         rank = retrieved_docs.index(actual_doc) + 1
        #         reciprocal_ranks.append(1.0 / rank)
        #     except ValueError:
        #         reciprocal_ranks.append(0.0)

        # mrr = np.mean(reciprocal_ranks)

        return RetrievalMetrics(accuracy_at_k=accuracy_at_k)
