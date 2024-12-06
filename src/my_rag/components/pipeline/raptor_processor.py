from .base import PipelineStep, PipelineData
import numpy as np
import umap
from sklearn.mixture import GaussianMixture
from typing import List, Dict, Optional, Any, Tuple
import pandas as pd
from ..llms.base import BaseLLM
from ..embeddings.base import BaseEmbedding


class RaptorProcessor(PipelineStep):
    """Pipeline step for RAPTOR (Recursive Abstractive Processing and Topical Organization for Retrieval)"""

    def __init__(
        self,
        llm: BaseLLM,
        embedding_model: BaseEmbedding,
        batch_size: int = 32,
        instruction: Optional[str] = None,
        dim: int = 10,
        threshold: float = 0.1,
        n_levels: int = 3,
        random_seed: int = 224,
        summarization_template: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
    ):
        self.llm = llm
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.instruction = instruction
        self.dim = dim
        self.threshold = threshold
        self.n_levels = n_levels
        self.random_seed = random_seed
        self.summarization_template = (
            summarization_template
            or """
            Here is a sub-set of documents. 
            Give a detailed summary of the content provided.
            
            Documents:
            {context}
            """
        )
        self.generation_config = generation_config or {}

    def global_cluster_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
        return umap.UMAP(
            n_neighbors=n_neighbors, n_components=self.dim, metric="cosine"
        ).fit_transform(embeddings)

    def local_cluster_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        return umap.UMAP(
            n_neighbors=10, n_components=self.dim, metric="cosine"
        ).fit_transform(embeddings)

    def get_optimal_clusters(
        self, embeddings: np.ndarray, max_clusters: int = 50
    ) -> int:
        max_clusters = min(max_clusters, len(embeddings))
        n_clusters = np.arange(1, max_clusters)
        bics = []
        for n in n_clusters:
            gm = GaussianMixture(n_components=n, random_state=self.random_seed)
            gm.fit(embeddings)
            bics.append(gm.bic(embeddings))
        return n_clusters[np.argmin(bics)]

    def GMM_cluster(self, embeddings: np.ndarray):
        n_clusters = self.get_optimal_clusters(embeddings)
        gm = GaussianMixture(n_components=n_clusters, random_state=self.random_seed)
        gm.fit(embeddings)
        probs = gm.predict_proba(embeddings)
        labels = [np.where(prob > self.threshold)[0] for prob in probs]
        return labels, n_clusters

    def perform_clustering(self, embeddings: np.ndarray) -> List[np.ndarray]:
        if len(embeddings) <= self.dim + 1:
            return [np.array([0]) for _ in range(len(embeddings))]

        reduced_embeddings_global = self.global_cluster_embeddings(embeddings)
        global_clusters, n_global_clusters = self.GMM_cluster(reduced_embeddings_global)

        all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
        total_clusters = 0

        for i in range(n_global_clusters):
            global_cluster_embeddings_ = embeddings[
                np.array([i in gc for gc in global_clusters])
            ]

            if len(global_cluster_embeddings_) == 0:
                continue
            if len(global_cluster_embeddings_) <= self.dim + 1:
                local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
                n_local_clusters = 1
            else:
                reduced_embeddings_local = self.local_cluster_embeddings(
                    global_cluster_embeddings_
                )
                local_clusters, n_local_clusters = self.GMM_cluster(
                    reduced_embeddings_local
                )

            for j in range(n_local_clusters):
                local_cluster_embeddings_ = global_cluster_embeddings_[
                    np.array([j in lc for lc in local_clusters])
                ]
                indices = np.where(
                    (embeddings == local_cluster_embeddings_[:, None]).all(-1)
                )[1]
                for idx in indices:
                    all_local_clusters[idx] = np.append(
                        all_local_clusters[idx], j + total_clusters
                    )

            total_clusters += n_local_clusters

        return all_local_clusters

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts"""
        text_embeddings = self.embedding_model.embed(
            texts, batch_size=self.batch_size, instruction=self.instruction
        )
        return np.array(text_embeddings)

    def fmt_txt(self, df: pd.DataFrame) -> str:
        """Format texts for summarization"""
        unique_txt = df["text"].tolist()
        return "\n--- --- \n--- --- \n".join(unique_txt)

    def embed_cluster_texts(self, texts: List[str]) -> pd.DataFrame:
        """Embed and cluster texts into initial DataFrame"""
        text_embeddings = self.embed_texts(texts)
        cluster_labels = self.perform_clustering(text_embeddings)

        return pd.DataFrame(
            {"text": texts, "embd": list(text_embeddings), "cluster": cluster_labels}
        )

    def embed_cluster_summarize_texts(
        self, texts: List[str], level: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process single level of embedding, clustering and summarization"""
        # Initial embedding and clustering
        df_clusters = self.embed_cluster_texts(texts)

        # Expand for cluster processing
        expanded_list = []
        for index, row in df_clusters.iterrows():
            for cluster in row["cluster"]:
                expanded_list.append(
                    {"text": row["text"], "embd": row["embd"], "cluster": cluster}
                )
        expanded_df = pd.DataFrame(expanded_list)

        # Get unique clusters
        all_clusters = expanded_df["cluster"].unique()
        print(f"--Generated {len(all_clusters)} clusters at level {level}--")

        # Generate summaries for each cluster
        summaries = []
        for cluster_id in all_clusters:
            df_cluster = expanded_df[expanded_df["cluster"] == cluster_id]
            formatted_txt = self.fmt_txt(df_cluster)
            summary = self.llm.generate_summary(formatted_txt, **self.generation_config)
            print(f"Cluster {cluster_id} Summary:", summary)
            summaries.append(summary)

        # Create summary DataFrame
        df_summary = pd.DataFrame(
            {
                "summaries": summaries,
                "level": [level] * len(summaries),
                "cluster": list(all_clusters),
            }
        )

        return df_clusters, df_summary

    def recursive_embed_cluster_summarize(
        self, texts: List[str], level: int = 1
    ) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Recursively process texts through levels of embedding, clustering and summarization"""
        results = {}

        # Process current level
        df_clusters, df_summary = self.embed_cluster_summarize_texts(texts, level)
        results[level] = (df_clusters, df_summary)

        # Check for further recursion
        unique_clusters = df_summary["cluster"].nunique()
        if level < self.n_levels and unique_clusters > 1:
            new_texts = df_summary["summaries"].tolist()
            next_level_results = self.recursive_embed_cluster_summarize(
                new_texts, level + 1
            )
            results.update(next_level_results)

        return results

    def run(self, pipeline_data: PipelineData) -> PipelineData:
        """Execute complete RAPTOR pipeline"""
        if not pipeline_data.documents:
            return pipeline_data

        results = self.recursive_embed_cluster_summarize(pipeline_data.documents)
        for level in sorted(results.keys()):
            summaries = results[level][1]["summaries"].tolist()
            pipeline_data.documents.extend(summaries)
            for i in range(len(summaries)):
                if pipeline_data.metadata:
                    pipeline_data.metadata.append(
                        {
                            "doc_id": f"summary_{i}_{level}",
                            "chunk_index": -1,
                            "total_chunks": -1,
                            "source_type": "summary",
                        }
                    )
        pipeline_data.embeddings = self.embed_texts(pipeline_data.documents)
        return pipeline_data
