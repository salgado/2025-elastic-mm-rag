import logging
from pathlib import Path
import os

from embedding_generator import EmbeddingGenerator
from elastic_manager import ElasticManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultimodalPipelineTester:
    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()
        self.elastic_manager = ElasticManager()
        
    def index_all_content(self, data_dir="data"):
        """Indexa todo o conteúdo do diretório data"""
        logger.info("🚀 Iniciando indexação de todo o conteúdo...")
        
        # Processa imagens
        image_dir = Path(data_dir) / "images"
        for img_path in image_dir.glob("*.jpg"):
            try:
                embedding = self.embedding_generator.generate_embedding([str(img_path)], "vision")
                self.elastic_manager.index_content(
                    embedding=embedding,
                    modality="vision",
                    description=f"Imagem: {img_path.name}",
                    content_path=str(img_path)
                )
                logger.info(f"✅ Indexada imagem: {img_path.name}")
            except Exception as e:
                logger.error(f"❌ Erro ao indexar imagem {img_path}: {str(e)}")
        
        # Processa áudios
        audio_dir = Path(data_dir) / "audios"
        for audio_path in audio_dir.glob("*.wav"):
            try:
                embedding = self.embedding_generator.generate_embedding([str(audio_path)], "audio")
                self.elastic_manager.index_content(
                    embedding=embedding,
                    modality="audio",
                    description=f"Áudio: {audio_path.name}",
                    content_path=str(audio_path)
                )
                logger.info(f"✅ Indexado áudio: {audio_path.name}")
            except Exception as e:
                logger.error(f"❌ Erro ao indexar áudio {audio_path}: {str(e)}")
        
        # Processa textos
        text_dir = Path(data_dir) / "texts"
        for text_path in text_dir.glob("*.txt"):
            try:
                with open(text_path, 'r') as f:
                    text = f.read()
                embedding = self.embedding_generator.generate_embedding([text], "text")
                self.elastic_manager.index_content(
                    embedding=embedding,
                    modality="text",
                    description=f"Texto: {text_path.name}",
                    content_path=str(text_path)
                )
                logger.info(f"✅ Indexado texto: {text_path.name}")
            except Exception as e:
                logger.error(f"❌ Erro ao indexar texto {text_path}: {str(e)}")
                
        # Processa depth maps
        depth_dir = Path(data_dir) / "depths"
        for depth_path in depth_dir.glob("*.jpg"):  # Procura por .jpg e .png
            try:
                embedding = self.embedding_generator.generate_embedding([str(depth_path)], "depth")
                self.elastic_manager.index_content(
                    embedding=embedding,
                    modality="depth",
                    description=f"Depth Map: {depth_path.name}",
                    content_path=str(depth_path)
                )
                logger.info(f"✅ Indexado depth map: {depth_path.name}")
            except Exception as e:
                logger.error(f"❌ Erro ao indexar depth map {depth_path}: {str(e)}")
    
    def test_cross_modal_search(self):
        """Testa busca cross-modal"""
        logger.info("\n🔍 Testando busca cross-modal...")
        
        # Busca por imagem
        test_image = "data/images/crime_scene1.jpg"
        logger.info(f"\nBuscando conteúdo similar à imagem: {test_image}")
        try:
            image_embedding = self.embedding_generator.generate_embedding([test_image], "vision")
            results = self.elastic_manager.search_similar(image_embedding)
            
            logger.info("Resultados encontrados:")
            for i, result in enumerate(results, 1):
                logger.info(f"{i}. {result['description']} ({result['modality']})")
        except Exception as e:
            logger.error(f"Erro na busca por imagem: {str(e)}")
        
        # Busca por áudio
        test_audio = "data/audios/joker_laugh.wav"
        logger.info(f"\nBuscando conteúdo similar ao áudio: {test_audio}")
        try:
            audio_embedding = self.embedding_generator.generate_embedding([test_audio], "audio")
            results = self.elastic_manager.search_similar(audio_embedding)
            
            logger.info("Resultados encontrados:")
            for i, result in enumerate(results, 1):
                logger.info(f"{i}. {result['description']} ({result['modality']})")
        except Exception as e:
            logger.error(f"Erro na busca por áudio: {str(e)}")
        
        # Busca por texto
        test_text = "Why so serious?"
        logger.info(f"\nBuscando conteúdo similar ao texto: {test_text}")
        try:
            text_embedding = self.embedding_generator.generate_embedding([test_text], "text")
            results = self.elastic_manager.search_similar(text_embedding)
            
            logger.info("Resultados encontrados:")
            for i, result in enumerate(results, 1):
                logger.info(f"{i}. {result['description']} ({result['modality']})")
        except Exception as e:
            logger.error(f"Erro na busca por texto: {str(e)}")
            
        # Busca por depth map
        test_depth = "data/depths/depth_map_result.png"
        logger.info(f"\nBuscando conteúdo similar ao depth map: {test_depth}")
        try:
            depth_embedding = self.embedding_generator.generate_embedding([test_depth], "depth")
            results = self.elastic_manager.search_similar(depth_embedding)
            
            logger.info("Resultados encontrados:")
            for i, result in enumerate(results, 1):
                logger.info(f"{i}. {result['description']} ({result['modality']})")
        except Exception as e:
            logger.error(f"Erro na busca por depth map: {str(e)}")

def generate_example_queries():
    """Gera exemplos de queries para o Kibana Dev Tools"""
    logger.info("\n📝 Exemplos de queries para Kibana Dev Tools:")
    
    # Query para criar o índice
    index_creation = """{
    "mappings": {
        "properties": {
            "embedding": {
                "type": "dense_vector",
                "dims": 1024,
                "index": true,
                "similarity": "cosine"
            },
            "modality": {"type": "keyword"},
            "content": {"type": "binary"},
            "description": {"type": "text"},
            "metadata": {"type": "object"},
            "content_path": {"type": "text"}
        }
    }
}"""
    
    logger.info(f"\n1. Criar índice:\nPUT multimodal_content\n{index_creation}")
    
    # Query para buscar por modalidade
    modality_query = """{
    "query": {
        "bool": {
            "must": [
                {"term": {"modality": "vision"}}
            ]
        }
    }
}"""
    
    logger.info(f"\n2. Buscar por modalidade:\nGET multimodal_content/_search\n{modality_query}")
    
    # Query para kNN search
    knn_query = """{
    "query": {
        "knn": {
            "embedding": {
                "vector": [0.1, 0.2, ...],  # Substitua com seu vetor
                "k": 5
            }
        }
    },
    "_source": ["content_path", "modality", "description"]
}"""
    
    logger.info(f"\n3. Busca kNN:\nGET multimodal_content/_search\n{knn_query}")

def main():
    logger.info("🎬 Iniciando teste do pipeline multimodal...")
    
    tester = MultimodalPipelineTester()
    
    # Indexa todo o conteúdo
    tester.index_all_content()
    
    # Testa buscas cross-modais
    tester.test_cross_modal_search()
    
    # Gera exemplos de queries
    generate_example_queries()
    
    logger.info("\n✨ Teste do pipeline concluído!")

if __name__ == "__main__":
    main()