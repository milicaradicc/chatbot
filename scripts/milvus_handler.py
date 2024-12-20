from typing import List
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
import numpy as np
from sentence_transformers import SentenceTransformer

class MilvusHandler:
    def __init__(self, host: str, port: str, model: SentenceTransformer, collection_name: str = 'chatbott'):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.collection = None
        self.model = model
        
    def connect(self):
        try:
            connections.connect(alias="default", host=self.host, port=self.port)
        except Exception as e:
            print(f"Milvus connection error: {e}")

    def create_collection(self):
        # Automatically generated id, original sentance and the embedding
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
            "params": {"nlist": 2048} # Higher values improve search speed but consume more memory
        }
        self.collection.create_index("embedding", index_params)

    def insert_embeddings(self, sentences: List[str], embeddings: np.ndarray):
        data = [
            sentences,  
            embeddings.tolist()  
        ]
        self.collection.insert(data)
        self.collection.flush()

    '''
    Total Clusters: 2048
    nprobe = 10  → Searches 10 clusters (fast, less precise)
    nprobe = 50  → Searches 50 clusters (moderate speed, balanced precision)
    nprobe = 200 → Searches 200 clusters (slow, high precision)
    '''
    def search_similar_sentences(self, query: str, top_k: int = 5) -> List[str]:
        self.collection.load()
        query_embedding = self.model.encode([query])[0]
        # Number of clusters
        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 50}
        }
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params, # Cos and clusters
            limit=top_k, # Max results
            output_fields=["sentence"] # Original sentance
        )
        return [hit.entity.get('sentence') for hit in results[0][:top_k]]

    