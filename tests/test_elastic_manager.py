# tests/test_elastic_manager.py

import unittest
import numpy as np
from src.elastic_manager import ElasticSearchManager
from src.config import ELASTIC_CONFIG

class TestElasticSearchManager(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests."""
        cls.es_manager = ElasticSearchManager(
            cloud_id=ELASTIC_CONFIG["cloud_id"],
            api_key=ELASTIC_CONFIG["api_key"],
            index_name="test_multimodal_content"
        )
        
        # Create test embedding
        cls.test_embedding = np.random.rand(1024).astype(np.float32)
        
    def setUp(self):
        """Clean up before each test."""
        # Delete any existing test data
        self.es_manager.delete_content(modality="test")
        
    def test_index_and_search(self):
        """Test indexing and searching content."""
        # Index a test document
        response = self.es_manager.index_content(
            embedding=self.test_embedding,
            modality="test",
            description="Test document",
            metadata={"test_id": 1}
        )
        
        self.assertIsNotNone(response)
        self.assertTrue("_id" in response)
        
        # Search for similar documents
        results = self.es_manager.search_content(
            query_embedding=self.test_embedding,
            k=1,
            modality="test"
        )
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["_source"]["description"], "Test document")
        
    def test_binary_content(self):
        """Test indexing and retrieving binary content."""
        # Create some binary content
        test_content = b"Hello, World!"
        
        # Index with binary content
        response = self.es_manager.index_content(
            embedding=self.test_embedding,
            modality="test",
            content=test_content,
            description="Binary test"
        )
        
        self.assertIsNotNone(response)
        
        # Retrieve and verify
        results = self.es_manager.search_content(
            query_embedding=self.test_embedding,
            k=1,
            modality="test"
        )
        
        self.assertEqual(len(results), 1)
        self.assertTrue("content" in results[0]["_source"])
        
    def test_modality_filter(self):
        """Test filtering by modality."""
        # Index documents with different modalities
        modalities = ["text", "vision", "audio"]
        
        for modality in modalities:
            self.es_manager.index_content(
                embedding=np.random.rand(1024),
                modality=modality,
                description=f"Test {modality}"
            )
            
        # Search with modality filter
        results = self.es_manager.search_content(
            query_embedding=np.random.rand(1024),
            k=5,
            modality="text"
        )
        
        self.assertTrue(all(hit["_source"]["modality"] == "text" for hit in results))
        
    def test_delete_content(self):
        """Test content deletion."""
        # Index a test document
        response = self.es_manager.index_content(
            embedding=self.test_embedding,
            modality="test",
            description="To be deleted"
        )
        
        doc_id = response["_id"]
        
        # Delete by ID
        success = self.es_manager.delete_content(doc_id=doc_id)
        self.assertTrue(success)
        
        # Verify deletion
        results = self.es_manager.search_content(
            query_embedding=self.test_embedding,
            k=1,
            modality="test"
        )
        
        self.assertEqual(len(results), 0)
        
    def test_invalid_connection(self):
        """Test handling of invalid connection."""
        with self.assertRaises(Exception):
            ElasticSearchManager(
                cloud_id="invalid_cloud_id",
                api_key="invalid_api_key"
            )

if __name__ == '__main__':
    unittest.main(verbosity=2)