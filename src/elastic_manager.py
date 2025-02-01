from elasticsearch import Elasticsearch, helpers
import base64
import os
from dotenv import load_dotenv
import numpy as np

class ElasticManager:
    """Gerencia operações multimodais no Elasticsearch"""
    
    def __init__(self):
        load_dotenv()  # Carrega variáveis do .env
        self.es = self._connect_elastic()
        self.index_name = "multimodal_content"
        self._setup_index()
    
    def _connect_elastic(self):
        """Conecta ao Elasticsearch"""
        return Elasticsearch(
            cloud_id=os.getenv("ELASTIC_CLOUD_ID"),
            api_key=os.getenv("ELASTIC_API_KEY")
        )
    
    def _setup_index(self):
        """Configura o índice se não existir"""
        if not self.es.indices.exists(index=self.index_name):
            mapping = {
                "mappings": {
                    "properties": {
                        "embedding": {
                            "type": "dense_vector",
                            "dims": 1024,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "modality": {"type": "keyword"},
                        "content": {"type": "binary"},
                        "description": {"type": "text"},
                        "metadata": {"type": "object"},
                        "content_path": {"type": "text"}
                    }
                }
            }
            self.es.indices.create(index=self.index_name, body=mapping)
    
    def index_content(self, embedding, modality, content=None, description="", metadata=None, content_path=None):
        """Indexa conteúdo multimodal"""
        doc = {
            "embedding": embedding.tolist(),
            "modality": modality,
            "description": description,
            "metadata": metadata or {},
            "content_path": content_path
        }
        
        if content:
            doc["content"] = base64.b64encode(content).decode() if isinstance(content, bytes) else content
        
        return self.es.index(index=self.index_name, document=doc)
    
    def search_similar(self, query_embedding, modality=None, k=5):
        """Busca conteúdos similares"""
        query = {
            "knn": {
                "field": "embedding",
                "query_vector": query_embedding.tolist(),
                "k": k,
                "num_candidates": 100,
                "filter": [{"term": {"modality": modality}}] if modality else []
            }
        }
        
        response = self.es.search(
            index=self.index_name,
            query=query,
            size=k,
            _source=["content_path", "modality", "description", "metadata"]
        )
        
        return [hit["_source"] for hit in response["hits"]["hits"]]