# Multimodal RAG with Elasticsearch - Gotham Detective

This project implements a Multimodal RAG (Retrieval-Augmented Generation) pipeline using:
- Elasticsearch for vector storage
- ImageBind for multimodal embeddings
- GPT-4 for analysis

## Project Structure

```
.
├── data/
│   ├── images/
│   ├── audios/
│   ├── texts/
│   └── depths/
└── src/
    ├── config.py
    ├── elastic_manager.py
    ├── imagebind_model.py
    ├── llm_integration.py
    └── main.py
```

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.template` to `.env` and fill in credentials
4. Run verification: `python src/verify_structure.py`

## Usage

[Documentation in progress]

