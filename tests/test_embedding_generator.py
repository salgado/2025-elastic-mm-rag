# tests/test_embedding_generator.py

import unittest
import numpy as np
from pathlib import Path
from src.embedding_generator import EmbeddingGenerator
from src.config import DIRECTORIES

class TestEmbeddingGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running all tests."""
        cls.generator = EmbeddingGenerator()
        cls.test_data_dir = Path(__file__).parent / 'data'
        cls.embedding_size = 1024  # Expected size of ImageBind embeddings

    def test_text_embedding(self):
        """Test generating embeddings for text."""
        text = "A mysterious night in Gotham City"
        embedding = self.generator.process_modality(text, "text")
        
        self.assertIsNotNone(embedding)
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape[0], self.embedding_size)

    def test_image_embedding(self):
        """Test generating embeddings for images."""
        test_image = DIRECTORIES['images'] / 'crime_scene.jpg'
        if not test_image.exists():
            self.skipTest(f"Test image not found: {test_image}")
            
        embedding = self.generator.process_modality(str(test_image), "vision")
        
        self.assertIsNotNone(embedding)
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape[0], self.embedding_size)

    def test_invalid_modality(self):
        """Test handling of invalid modality."""
        embedding = self.generator.process_modality("test", "invalid_modality")
        self.assertIsNone(embedding)

    def test_missing_file(self):
        """Test handling of missing files."""
        embedding = self.generator.process_modality(
            "nonexistent_file.jpg", "vision"
        )
        self.assertIsNone(embedding)

if __name__ == '__main__':
    unittest.main(verbosity=2)