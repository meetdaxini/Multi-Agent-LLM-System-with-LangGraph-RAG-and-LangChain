from .base import BaseVectorStore
from typing import Any, List, Optional, Dict
import deeplake
import numpy as np
import os

class DeepLakeVectorStore(BaseVectorStore):
    """
    Vector store implementation using Deep Lake.
    """

    def __init__(self, dataset_path: str, overwrite: bool = False):
        """
        Initializes the Deep Lake vector store.

        Args:
            dataset_path (str): The path to the Deep Lake dataset.
            overwrite (bool): Whether to overwrite the existing dataset.
        """
        self.dataset_path = dataset_path

        if overwrite and os.path.exists(dataset_path):
            deeplake.delete(dataset_path)

        try:
            self.ds = deeplake.load(dataset_path)
        except Exception:
            self.ds = deeplake.empty(dataset_path)
            self.ds.create_tensor('embedding', htype='generic')
            self.ds.create_tensor('metadata', htype='json')

    def add_embeddings(self, embeddings: Any, metadata: Optional[List[dict]] = None):
        """
        Adds embeddings and metadata to the vector store.

        Args:
            embeddings (Any): The embeddings to add (NumPy array).
            metadata (Optional[List[dict]]): Optional metadata for each embedding.
        """
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        num_embeddings = embeddings.shape[0]
        if metadata is None:
            metadata = [{}] * num_embeddings
        elif len(metadata) != num_embeddings:
            raise ValueError("Length of metadata does not match number of embeddings.")

        self.ds.embedding.append(embeddings)
        self.ds.metadata.append(metadata)
        self.ds.commit("Added new embeddings.")

    def search(self, query_embedding: Any, k: int) -> List[Dict]:
        """
        Searches for the top k most similar embeddings.

        Args:
            query_embedding (Any): The embedding of the query (NumPy array).
            k (int): The number of top results to return.

        Returns:
            List[Dict]: A list of results with 'embedding' and 'metadata'.
        """
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)

        embeddings = np.array(self.ds.embedding)
        similarities = embeddings @ query_embedding.T
        top_k_indices = np.argsort(-similarities.flatten())[:k]
        results = []
        for idx in top_k_indices:
            result = {
                'embedding': embeddings[0][idx],
                'metadata': self.ds.metadata[0].data()['value'][idx]
            }
            results.append(result)

        return results

    # TODO: def backup(self, path: str):
    #     """
    #     backups the vector store to a file.

    #     Args:
    #         path (str): The file path to backup the vector store.
    #     """
    #     deeplake.copy(self.dataset_path, path)

    def load(self, path: str):
        """
        Loads the vector store from a file.

        Args:
            path (str): The file path to load the vector store from.
        """
        self.ds = deeplake.load(path)

    def clean_up(self):
        """
        Cleans up resources to free memory.
        """
        # TODO: deep lake clean up
        # self.ds.delete_by_path(self.dataset_path)
        deeplake.delete(self.dataset_path)
