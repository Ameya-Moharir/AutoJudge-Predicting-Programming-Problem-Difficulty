#!/bin/bash

# AutoJudge Setup Script
# This script sets up the environment and trains models

echo "================================================"
echo "AutoJudge - Setup and Training Script"
echo "================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo -e "${RED}Error: Python 3.8 or higher is required${NC}"
    echo "Current version: $python_version"
    exit 1
fi

echo -e "${GREEN}✓ Python version OK: $python_version${NC}"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}Virtual environment already exists${NC}"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1

# Install dependencies
echo "Installing dependencies..."
echo "This may take a few minutes..."
pip install -r requirements.txt > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Dependencies installed successfully${NC}"
else
    echo -e "${RED}Error installing dependencies${NC}"
    exit 1
fi
echo ""

# Download NLTK data
echo "Downloading NLTK data..."
python3 -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('punkt', quiet=True); nltk.download('wordnet', quiet=True)" > /dev/null 2>&1
echo -e "${GREEN}✓ NLTK data downloaded${NC}"
echo ""

# Train models
echo "================================================"
echo "Training Models"
echo "================================================"
echo ""
echo "Choose training option:"
echo "1) Quick training (sample data, ~1 minute)"
echo "2) Full training (download datasets, ~5 minutes)"
echo ""
read -p "Enter choice (1 or 2): " choice

if [ "$choice" = "1" ]; then
    echo ""
    echo "Starting quick training..."
    python3 scripts/train_models.py --quick
elif [ "$choice" = "2" ]; then
    echo ""
    echo "Starting full training..."
    python3 scripts/train_models.py --full
else
    echo -e "${YELLOW}Invalid choice. Running quick training...${NC}"
    python3 scripts/train_models.py --quick
fi

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Training completed successfully!${NC}"
    echo ""
    echo "================================================"
    echo "Setup Complete!"
    echo "================================================"
    echo ""
    echo "To start the web application, run:"
    echo ""
    echo "  source venv/bin/activate"
    echo "  python3 web/app.py"
    echo ""
    echo "Then open http://localhost:5000 in your browser"
    echo ""
else
    echo -e "${RED}Error during training${NC}"
    exit 1
fi
