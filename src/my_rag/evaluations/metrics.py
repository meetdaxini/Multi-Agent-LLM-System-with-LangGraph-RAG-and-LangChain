from dataclasses import dataclass
from typing import List, Dict
import numpy as np


@dataclass
class RetrievalMetrics:
    """Class to hold retrieval evaluation metrics"""

    accuracy_at_k: Dict[int, float]
    mrr: float

    def to_dict(self) -> Dict[str, float]:
        metrics = {}
        for k, acc in self.accuracy_at_k.items():
            metrics[f"Accuracy@{k}"] = acc

        metrics["MRR"] = self.mrr
        return metrics


class MetricsCalculator:
    """Calculator for retrieval metrics"""

    @staticmethod
    def calculate_metrics(
        retrieved_doc_ids: List[List[str]], actual_doc_ids: List[str], max_k: int
    ) -> RetrievalMetrics:
        accuracy_at_k = {}
        reciprocal_ranks = []

        total_queries = len(actual_doc_ids)

        for k in range(1, max_k + 1):
            correct_at_k = 0

            for retrieved_docs, actual_doc in zip(retrieved_doc_ids, actual_doc_ids):
                # Get unique doc IDs up to k
                top_k_docs = list(retrieved_docs[:k])

                # Accuracy
                if actual_doc in top_k_docs:
                    correct_at_k += 1

            accuracy_at_k[k] = correct_at_k / total_queries

        # Calculate MRR
        for retrieved_docs, actual_doc in zip(retrieved_doc_ids, actual_doc_ids):
            try:
                rank = retrieved_docs.index(actual_doc) + 1
                reciprocal_ranks.append(1.0 / rank)
            except ValueError:
                reciprocal_ranks.append(0.0)

        mrr = np.mean(reciprocal_ranks)

        return RetrievalMetrics(accuracy_at_k=accuracy_at_k, mrr=mrr)
