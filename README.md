# Multimodal RAG Pipeline with Elasticsearch

A multimodal Retrieval-Augmented Generation (RAG) pipeline using Elasticsearch to solve the mystery in Gotham City.

## 🌟 Overview

The system processes and analyzes different types of evidence:
- 🖼️ Crime scene images
- 🔊 Audio recordings
- 📝 Text messages and notes
- 📊 Depth maps

## 🏗️ Project Structure

```
2025-elastic-mm-rag/
├── src/                      # Main source code
│   ├── pipeline.py          # Main processing pipeline
│   ├── embedding_generator.py # ImageBind embedding generation
│   ├── elastic_manager.py    # Elasticsearch interface
│   └── llm_analyzer.py      # GPT-4 analysis
│
├── tests/                    # Automated tests
│   ├── test_elastic_manager.py
│   ├── test_embedding_generator.py
│   ├── test_llm_analyzer.py
│   ├── test_pipeline.py
│   └── test_utils.py
│
└── data/                     # Sample data
    ├── images/              # Case images
    ├── audios/              # Audio recordings
    ├── texts/               # Text messages
    └── depths/              # Depth maps
```

## 🚀 Getting Started

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/salgado/2025-elastic-mm-rag.git
cd 2025-elastic-mm-rag
```

2. Create a virtual environment:
```bash
conda create -n env_blog_mmrag python=3.8
conda activate env_blog_mmrag
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.template .env
# Edit .env with your credentials
```

### Running the Pipeline

1. Complete pipeline:
```bash
python src/pipeline.py
```

2. Specific tests:
```bash
# Test embedding generator
python tests/test_embedding_generator.py

# Test LLM analyzer
python tests/test_llm_analyzer.py
```

## 🧩 Key Components

### EmbeddingGenerator
- Uses ImageBind for multimodal embedding generation
- Supports images, audio, text, and depth maps
- Generates 1024-dimensional vectors

### ElasticManager
- Manages Elasticsearch connections
- Stores and retrieves embeddings
- Implements similarity search

### LLMAnalyzer
- Uses GPT-4 for forensic analysis
- Generates detailed reports
- Analyzes connections between different types of evidence

## 📝 Usage Example

Example of evidence analysis:
```python
from src.embedding_generator import EmbeddingGenerator
from src.elastic_manager import ElasticManager
from src.llm_analyzer import LLMAnalyzer

# Generate embedding
generator = EmbeddingGenerator()
embedding = generator.generate_embedding(["data/images/crime_scene1.jpg"], "vision")

# Search for similar evidence
elastic = ElasticManager()
results = elastic.search_similar(embedding)

# Analyze results
analyzer = LLMAnalyzer()
report = analyzer.analyze_evidence(results)
```

## 🛠️ System Requirements

- Python 3.8 or higher
- Elasticsearch 8.11.0
- GPU recommended for faster processing

## 📄 License

This project is licensed under the MIT License.

## 🙋‍♂️ Support

For questions and support:
1. Open an issue on GitHub

## 🔗 Useful Links

- [ImageBind Documentation](https://github.com/facebookresearch/ImageBind)
- [Elasticsearch Documentation](https://www.elastic.co/guide/index.html)
- [OpenAI GPT-4](https://platform.openai.com/docs/models)
