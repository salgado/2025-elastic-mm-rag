# src/verify_structure.py
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Directory base for evidence files
DATA_DIR = "data"

# List of expected files and directories
evidence = {
    "images": ["crime_scene_1.jpg", "suspect_spotted.jpg"],
    "audios": ["joker_laugh.wav"],
    "texts": ["riddle.txt"],
    "depths": ["depth_suspect.png"]
}

def verify_structure():
    """Verify the presence of all required directories and files."""
    logger.info("🔍 Verifying project structure...")
    
    # First verify the base data directory
    if not os.path.exists(DATA_DIR):
        logger.error(f"❌ Base data directory missing: {DATA_DIR}")
        return False
    
    all_valid = True
    
    # Check each category directory and its files
    for category, files in evidence.items():
        category_path = os.path.join(DATA_DIR, category)
        
        # Check directory existence
        if not os.path.exists(category_path):
            logger.error(f"❌ Missing directory: {category_path}")
            all_valid = False
            continue
        
        logger.info(f"✅ Found directory: {category_path}")
        
        # Check README.md in each directory
        readme_path = os.path.join(category_path, "README.md")
        if not os.path.exists(readme_path):
            logger.warning(f"⚠️  Missing README.md in {category_path}")
        
        # Check for required files
        for file in files:
            file_path = os.path.join(category_path, file)
            if not os.path.exists(file_path):
                logger.warning(f"⚠️  Warning: {file} not found in {category_path}")
                all_valid = False
            else:
                logger.info(f"✅ Found file: {file}")
    
    return all_valid

if __name__ == "__main__":
    if verify_structure():
        logger.info("✅ Project structure verification completed successfully!")
    else:
        logger.warning("⚠️  Project structure verification completed with warnings.")