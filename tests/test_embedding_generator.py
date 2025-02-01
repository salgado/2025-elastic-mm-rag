import numpy as np
import logging
from pathlib import Path
import sys
import os

# Adiciona o diretório src ao PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestEmbeddingGenerator:
    def __init__(self):
        try:
            from src.embedding_generator import EmbeddingGenerator
            self.generator = EmbeddingGenerator()
            logger.info("EmbeddingGenerator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EmbeddingGenerator: {e}")
            raise

    def test_image_embedding(self, image_path="data/images/crime_scene1.jpg"):
        """Test image embedding generation"""
        try:
            embedding = self.generator.generate_embedding([image_path], "vision")
            logger.info(f"Image embedding shape: {embedding.shape}")
            logger.info(f"Image embedding stats - Mean: {np.mean(embedding):.4f}, Std: {np.std(embedding):.4f}")
            return embedding
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return None

    def test_audio_embedding(self, audio_path="data/audios/joker_laugh.wav"):
        """Test audio embedding generation"""
        try:
            embedding = self.generator.generate_embedding([audio_path], "audio")
            logger.info(f"Audio embedding shape: {embedding.shape}")
            logger.info(f"Audio embedding stats - Mean: {np.mean(embedding):.4f}, Std: {np.std(embedding):.4f}")
            return embedding
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return None

    def test_text_embedding(self, text_path="data/texts/riddle.txt"):
        """Test text embedding generation"""
        try:
            with open(text_path, 'r') as f:
                text = f.read()
            embedding = self.generator.generate_embedding([text], "text")
            logger.info(f"Text embedding shape: {embedding.shape}")
            logger.info(f"Text embedding stats - Mean: {np.mean(embedding):.4f}, Std: {np.std(embedding):.4f}")
            return embedding
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            return None

    def test_depth_embedding(self, depth_path="data/depths/depth_map_result.png"):
        """Test depth map embedding generation"""
        try:
            logger.info(f"Processing depth map: {depth_path}")
            
            # Verificar se o arquivo existe
            if not Path(depth_path).exists():
                raise FileNotFoundError(f"Depth map file not found: {depth_path}")
                
            embedding = self.generator.generate_embedding([depth_path], "depth")
            logger.info(f"Depth embedding shape: {embedding.shape}")
            logger.info(f"Depth embedding stats - Mean: {np.mean(embedding):.4f}, Std: {np.std(embedding):.4f}")
            return embedding
        except Exception as e:
            logger.error(f"Error processing depth map: {str(e)}", exc_info=True)
            return None
            
def main():
    logger.info("🚀 Starting embedding generator tests...")
    
    tester = TestEmbeddingGenerator()
    
    # Test each modality
    logger.info("\n🖼️ Testing image embedding...")
    image_emb = tester.test_image_embedding()
    
    logger.info("\n🔊 Testing audio embedding...")
    audio_emb = tester.test_audio_embedding()
    
    logger.info("\n📝 Testing text embedding...")
    text_emb = tester.test_text_embedding()
    
    logger.info("\n📊 Testing depth embedding...")
    depth_emb = tester.test_depth_embedding()
    
    # Check if all embeddings have the same dimensionality
    embeddings = [e for e in [image_emb, audio_emb, text_emb, depth_emb] if e is not None]
    if embeddings:
        shapes = [e.shape for e in embeddings]
        logger.info(f"\n📐 All generated embedding shapes: {shapes}")
        if len(set(shapes)) == 1:
            logger.info("✅ All embeddings have the same dimensionality")
        else:
            logger.warning("⚠️ Embeddings have different dimensionalities")

if __name__ == "__main__":
    main()