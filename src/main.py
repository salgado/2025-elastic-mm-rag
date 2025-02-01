from embedding_generator import EmbeddingGenerator
from elastic_manager import ElasticManager
import os

# Configuração
device = "cuda" if torch.cuda.is_available() else "cpu"

# Inicializa componentes
embedder = EmbeddingGenerator(device=device)
es_manager = ElasticManager()

# Exemplo de indexação
def index_example():
    content = {
        "path": "caminho/da/imagem.jpg",
        "modality": "vision",
        "description": "Descrição do conteúdo"
    }
    
    embedding = embedder.generate_embedding(
        input_data=content["path"],
        modality=content["modality"]
    )
    
    with open(content["path"], "rb") as f:
        content_bytes = f.read()
    
    es_manager.index_content(
        embedding=embedding,
        modality=content["modality"],
        content=content_bytes,
        description=content["description"],
        content_path=content["path"]
    )

# Exemplo de busca
def search_example():
    query_embedding = embedder.generate_embedding(
        input_data="texto de consulta",
        modality="text"
    )
    
    results = es_manager.search_similar(
        query_embedding=query_embedding,
        modality="vision",
        k=5
    )
    
    for result in results:
        print(f"Resultado: {result['description']}")

if __name__ == "__main__":
    index_example()
    search_example()