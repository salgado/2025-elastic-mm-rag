# 🕵️ Multimodal RAG with Elasticsearch - Gotham Detective

[![Open in GitHub](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/salgado/2025-elastic-mm-rag)

This project implements a Multimodal RAG (Retrieval-Augmented Generation) pipeline using:

- **Elasticsearch** for vector storage and search
- **ImageBind** for multimodal embeddings (text, images, audio, depth)
- **GPT-4** for forensic analysis and report generation

## 🚀 Getting Started

### Prerequisites
- Conda/Miniconda installed
- Python 3.10+
- Git LFS (for sample data if used)
- Elasticsearch Cloud account or local instance

### Project Structure
```
.
├── data/
│   ├── images/       # Crime scene images
│   ├── audios/       # Audio evidence
│   ├── texts/        # Text clues
│   └── depths/       # Depth maps
├── src/
│   ├── config.py     # Environment configuration
│   ├── elastic_manager.py  # Elasticsearch operations
│   ├── embedding_generator.py  # ImageBind integration
│   ├── llm_analyzer.py  # GPT-4 integration
│   └── main.py       # Main pipeline
└── requirements.txt  # Python dependencies
```

### 🔧 Installation

1. **Clone Repository**
   ```bash
   git clone https://github.com/salgado/2025-elastic-mm-rag.git
   cd 2025-elastic-mm-rag
   ```

2. **Create Conda Environment**
   ```bash
   conda create -n mmrag python=3.10 -y
   conda activate mmrag
   ```

3. **Install Dependencies**
   ```bash
   # Core dependencies
   pip install -r requirements.txt

   # ImageBind (required separately)
   pip install "git+https://github.com/facebookresearch/ImageBind.git@main#egg=imagebind"
   ```

4. **Environment Configuration**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` with your credentials:
   ```ini
   ELASTIC_CLOUD_ID="your_cloud_id"
   ELASTIC_API_KEY="your_api_key"
   OPENAI_API_KEY="sk-your-openai-key"
   ```

5. **Prepare Data Directories**
   ```bash
   mkdir -p data/{images,audios,texts,depths}
   # Add your evidence files to corresponding directories
   ```

6. **Verify Installation**
   ```bash
   python src/verify_structure.py  # Checks directory structure
   python -c "from imagebind.models import imagebind_model; print('✅ ImageBind ready!')"
   ```

### 🕵️ Usage

**Start Investigation**
```bash
python src/main.py --evidence data/audios/joker_laugh.wav --modality audio
```

**Key Features**
```bash
# Index evidence
python src/main.py --index --path data/images/crime_scene_1.jpg

# Search similar evidence
python src/main.py --search "data/audios/suspicious_laugh.wav" --k 5

# Generate forensic report
python src/main.py --analyze --output report.md
```

### 🚨 Troubleshooting

**Common Issues**
- **ImageBind Installation Failures**
  ```bash
  rm -rf ~/.cache/torch/hub/  # Clear corrupted cache
  pip install --force-reinstall imagebind
  ```
  
- **Elasticsearch Connection Issues**
  ```bash
  docker-compose restart elasticsearch  # If using local instance
  # Verify credentials in .env
  ```

[![Documentation](https://img.shields.io/badge/Docs-In_Progress-yellow)](CONTRIBUTING.md)
```

This version:
1. Uses clearer emoji-based section headers
2. Adds status badges
3. Provides complete setup instructions
4. Includes example commands
5. Maintains project structure visualization
6. Adds troubleshooting section
7. Links to future documentation

Would you like me to add any specific section or modify the tone? 🛠️
