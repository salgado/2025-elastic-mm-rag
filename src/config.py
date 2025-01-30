import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Ensure data directories exist
DIRECTORIES = {
    "images": DATA_DIR / "images",
    "audios": DATA_DIR / "audios",
    "texts": DATA_DIR / "texts",
    "depths": DATA_DIR / "depths"
}

# Create directories if they don't exist
for dir_path in DIRECTORIES.values():
    dir_path.mkdir(parents=True, exist_ok=True)

# Elasticsearch configuration
ELASTIC_CONFIG = {
    "cloud_id": os.getenv("ELASTIC_CLOUD_ID"),
    "api_key": os.getenv("ELASTIC_API_KEY"),
    "index_name": "gotham_evidence"
}

# OpenAI configuration
OPENAI_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY"),
    "model": "gpt-4-turbo-preview"
}

def verify_environment():
    """Verify that all required environment variables are set."""
    required_vars = {
        "ELASTIC_CLOUD_ID": ELASTIC_CONFIG["cloud_id"],
        "ELASTIC_API_KEY": ELASTIC_CONFIG["api_key"],
        "OPENAI_API_KEY": OPENAI_CONFIG["api_key"]
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    
    if missing_vars:
        logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
        logger.error("Please set these variables in your .env file")
        return False
        
    return True

def get_file_path(category: str, filename: str) -> Path:
    """Get the full path for a file in a specific category directory."""
    if category not in DIRECTORIES:
        raise ValueError(f"Invalid category: {category}. Must be one of {list(DIRECTORIES.keys())}")
    
    return DIRECTORIES[category] / filename

def configure_hardware():
    """Configure hardware settings (GPU/CPU) for PyTorch."""
    import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    if device == "cuda":
        logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    return device