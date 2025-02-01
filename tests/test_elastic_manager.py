import logging
import numpy as np
import sys
import os
from pathlib import Path

# Adiciona o diretório src ao PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestElasticManager:
    def __init__(self):
        try:
            from src.elastic_manager import ElasticManager
            from src.embedding_generator import EmbeddingGenerator
            
            self.elastic = ElasticManager()
            self.embedding_generator = EmbeddingGenerator()
            logger.info("✅ ElasticManager and EmbeddingGenerator initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize managers: {e}")
            raise

    def test_index_and_search(self):
        """Test indexing and searching content"""
        try:
            # Test with image
            image_path = "data/images/crime_scene1.jpg"
            logger.info(f"\n🔍 Testing with image: {image_path}")
            
            # Generate embedding
            image_embedding = self.embedding_generator.generate_embedding([image_path], "vision")
            
            # Index content
            self.elastic.index_content(
                embedding=image_embedding,
                modality="vision",
                description="Test image of crime scene",
                content_path=image_path
            )
            logger.info("✅ Image content indexed successfully")
            
            # Search similar content
            results = self.elastic.search_similar(image_embedding, k=5)
            logger.info(f"Found {len(results)} similar items")
            
            for i, result in enumerate(results, 1):
                logger.info(f"{i}. {result['description']} ({result['modality']})")
                
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in index and search test: {e}")
            return False

    def test_multiple_modalities(self):
        """Test handling multiple modalities"""
        try:
            test_files = {
                "vision": "data/images/crime_scene1.jpg",
                "audio": "data/audios/joker_laugh.wav",
                "text": "data/texts/riddle.txt"
            }
            
            for modality, file_path in test_files.items():
                logger.info(f"\n🔍 Testing {modality} modality with {file_path}")
                
                if modality == "text":
                    with open(file_path, 'r') as f:
                        content = f.read()
                    embedding = self.embedding_generator.generate_embedding([content], modality)
                else:
                    embedding = self.embedding_generator.generate_embedding([file_path], modality)
                
                # Index content
                self.elastic.index_content(
                    embedding=embedding,
                    modality=modality,
                    description=f"Test {modality} content",
                    content_path=file_path
                )
                logger.info(f"✅ {modality.capitalize()} content indexed successfully")
                
                # Search similar content
                results = self.elastic.search_similar(embedding, modality=modality, k=3)
                logger.info(f"Found {len(results)} similar {modality} items")
                
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in multiple modalities test: {e}")
            return False

def main():
    logger.info("🚀 Starting ElasticManager tests...")
    
    tester = TestElasticManager()
    
    # Run tests
    logger.info("\n📝 Testing basic index and search functionality...")
    index_search_success = tester.test_index_and_search()
    
    logger.info("\n📝 Testing multiple modalities...")
    multi_modal_success = tester.test_multiple_modalities()
    
    # Report results
    logger.info("\n📊 Test Results:")
    logger.info(f"Basic Index/Search: {'✅' if index_search_success else '❌'}")
    logger.info(f"Multiple Modalities: {'✅' if multi_modal_success else '❌'}")
    
    if index_search_success and multi_modal_success:
        logger.info("\n✨ All tests passed successfully!")
    else:
        logger.error("\n❌ Some tests failed")

if __name__ == "__main__":
    main()