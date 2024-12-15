from typing import List
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
import numpy as np
from sentence_transformers import SentenceTransformer
class MilvusHandler:
    def __init__(self, host: str, port: str, model: SentenceTransformer, collection_name: str = 'efdfdvjdb'):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.collection = None
        self.model = model
        
    def connect(self):
        """Connect to Milvus server."""
        try:
            connections.connect(alias="default", host=self.host, port=self.port)
        except Exception as e:
            print(f"Milvus connection error: {e}")

    def create_collection(self):
        """Create a Milvus collection."""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="sentence", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
        ]
        schema = CollectionSchema(fields, description="Wikipedia Sentences Collection")
        self.collection = Collection(name=self.collection_name, schema=schema)
        
        # IP (Inner Product): Used to measure cosine similarity (larger inner product indicates more similarity).
        # cosine_similarity — scikit-learn 1.5.2 documentation
        index_params = {
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 2048} # higher values improve search speed but consume more memory
        }
        self.collection.create_index("embedding", index_params)

    def insert_embeddings(self, sentences: List[str], embeddings: np.ndarray):
        data = [
            sentences,  
            embeddings.tolist()  
        ]
        self.collection.insert(data)
        self.collection.flush()

    def search_similar_sentences(self, query: str, top_k: int = 5) -> List[str]:
        self.collection.load()
        query_embedding = self.model.encode([query])[0]
        # euclidean distance
        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 50}
        }
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["sentence"]
        )
        unique_sentences = []
        seen = set()
        for hit in results[0]:
            sentence = hit.entity.get('sentence')
            if sentence not in seen:
                unique_sentences.append(sentence)
                seen.add(sentence)
                if len(unique_sentences) == top_k:
                    break
        
        return unique_sentences
    

        # recenice i njihove pozicije
        # similar_sentences = [hit.entity.get('sentence') for hit in results[0]]
        # sentence_positions = [hit.entity.get('position') for hit in results[0]]
        
        # window_sentences = []
        
        # komsije u window
        # for position in sentence_positions:
        #     start = max(0, position - window_size)
        #     end = min(len(self.document_sentences), position + window_size + 1)
        #     window_sentences.extend(self.document_sentences[start:end])
        
