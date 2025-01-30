#!/bin/bash

echo "🚀 Starting project setup with Conda..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install Conda first."
    exit 1
fi

# Create environment.yml file
cat > environment.yml << EOL
name: gotham-detective
channels:
  - conda-forge
  - pytorch
  - defaults
dependencies:
  - python=3.10
  - pip
  - pytorch
  - torchvision
  - cudatoolkit
  - pip:
    - elasticsearch==8.11.0
    - transformers>=4.30.0
    - openai>=1.3.0
    - python-dotenv>=0.19.0
    - Pillow>=9.0.0
    - numpy>=1.21.0
    - pandas>=1.3.0
    - tqdm>=4.65.0
    - python-magic>=0.4.27
EOL

# Remove existing environment if it exists
conda env remove -n gotham-detective

# Create conda environment
echo "🔧 Creating conda environment..."
conda env create -f environment.yml

# Activate environment
echo "🔄 Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate gotham-detective

if [ $? -eq 0 ]; then
    echo "✅ Conda environment activated successfully!"
else
    echo "❌ Failed to activate conda environment"
    echo "Try running: conda activate gotham-detective manually"
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    cp .env.template .env
    echo "✅ Created .env file. Please update it with your credentials."
fi

# Run structure verification
echo "🔍 Verifying project structure..."
python src/verify_structure.py

echo "
🎉 Setup complete! Next steps:
1. Edit .env file with your credentials
2. To activate the environment in new terminals, run:
   conda activate gotham-detective

Note: You might need to restart your terminal and run:
conda activate gotham-detective
"