from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class Visualizer:
    @staticmethod
    def visualize_embeddings_pca(embeddings: np.ndarray, sentences: Optional[List[str]] = None, n_components: int = 2):
        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(embeddings)

        plt.figure(figsize=(12, 10))
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.7)

        if sentences is not None:
            np.random.seed(42)
            sample_indices = np.random.choice(len(sentences), min(10, len(sentences)), replace=False)
            for idx in sample_indices:
                plt.annotate(sentences[idx][:50] + '...', 
                             (reduced_embeddings[idx, 0], reduced_embeddings[idx, 1]), 
                             fontsize=8, alpha=0.7)

        plt.title('Sentence Embeddings - PCA Visualization')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.tight_layout()
        plt.show()
