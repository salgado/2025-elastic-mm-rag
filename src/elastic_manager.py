# src/elastic_manager.py

from elasticsearch import Elasticsearch
import base64
import logging

logger = logging.getLogger(__name__)

class ElasticSearchManager:
    def __init__(self, cloud_id, api_key, index_name="multimodal_content"):
        """Initialize Elasticsearch manager.
        
        Args:
            cloud_id: Elasticsearch Cloud ID
            api_key: Elasticsearch API Key
            index_name: Name of the index (default: multimodal_content)
        """
        self.index_name = index_name
        self.es = self._create_client(cloud_id, api_key)
        self._setup_index()

    def _create_client(self, cloud_id, api_key):
        """Create and verify Elasticsearch connection."""
        try:
            client = Elasticsearch(
                cloud_id=cloud_id,
                api_key=api_key,
                request_timeout=30
            )
            if not client.ping():
                raise ConnectionError("Failed to connect to Elasticsearch")
            logger.info("🔗 Connected to Elasticsearch")
            return client
        except Exception as e:
            logger.error(f"🚨 Connection error: {str(e)}")
            raise

    def _setup_index(self):
        """Set up index if it doesn't exist."""
        try:
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
                            "metadata": {"type": "object"},
                            "description": {"type": "text"},
                            "content_path": {"type": "keyword"}
                        }
                    }
                }
                self.es.indices.create(index=self.index_name, body=mapping)
                logger.info(f"📂 Index {self.index_name} created successfully")
        except Exception as e:
            logger.error(f"🚨 Error creating index: {str(e)}")
            raise

    def index_content(self, embedding, modality, content=None, metadata=None, **kwargs):
        """Index a multimedia document with its embedding.
        
        Args:
            embedding: numpy array of the content embedding
            modality: type of content ('vision', 'text', 'audio', etc)
            content: binary content to be indexed (optional)
            metadata: additional metadata dictionary (optional)
            **kwargs: additional fields to index
            
        Returns:
            dict: Elasticsearch response or None if indexing fails
        """
        try:
            # Prepare the document
            doc = {
                "embedding": embedding.tolist(),
                "modality": modality,
                "metadata": metadata or {},
                **kwargs
            }

            # Add binary content if provided
            if content:
                encoded = base64.b64encode(content).decode().rstrip('=')
                doc["content"] = encoded

            # Index the document
            response = self.es.index(
                index=self.index_name,
                document=doc,
                refresh=True  # Make document immediately searchable
            )
            
            logger.info(f"📤 Successfully indexed document with ID: {response['_id']}")
            return response
            
        except Exception as e:
            logger.error(f"📤 Error indexing content: {str(e)}")
            return None

    def search_content(self, query_embedding, k=5, modality=None):
        """Perform similarity search for content.
        
        Args:
            query_embedding: numpy array to search for
            k: number of results to return (default: 5)
            modality: filter by content type (optional)
            
        Returns:
            list: List of similar documents with their scores
        """
        try:
            # Prepare kNN query
            knn = {
                "field": "embedding",
                "query_vector": query_embedding.tolist(),
                "k": k,
                "num_candidates": 100
            }

            # Add modality filter if specified
            query = {"knn": knn}
            if modality:
                query = {
                    "bool": {
                        "must": [
                            query,
                            {"term": {"modality": modality}}
                        ]
                    }
                }

            # Execute search
            response = self.es.search(
                index=self.index_name,
                query=query,
                source_includes=["content", "modality", "description", "metadata", "content_path"],
                size=k
            )
            
            results = response['hits']['hits']
            logger.info(f"🔍 Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"🔍 Search error: {str(e)}")
            return []
            
    def delete_content(self, doc_id=None, modality=None):
        """Delete content from the index.
        
        Args:
            doc_id: specific document ID to delete (optional)
            modality: delete all documents of this modality (optional)
            
        Returns:
            bool: True if deletion was successful
        """
        try:
            if doc_id:
                self.es.delete(index=self.index_name, id=doc_id, refresh=True)
                logger.info(f"🗑️ Deleted document {doc_id}")
                return True
                
            elif modality:
                response = self.es.delete_by_query(
                    index=self.index_name,
                    query={"term": {"modality": modality}},
                    refresh=True
                )
                logger.info(f"🗑️ Deleted {response['deleted']} documents with modality {modality}")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"🗑️ Error deleting content: {str(e)}")
            return False