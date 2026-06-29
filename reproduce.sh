#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status
set -e

echo "=== CRISP-DM Reproducibility Runner ==="
echo "This script automates setting up the environment and running the XAI benchmarking pipeline."

# 1. Create a local virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment 'venv'..."
    python3 -m venv venv
else
    echo "Virtual environment 'venv' already exists."
fi

# 2. Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# 3. Upgrade pip and install requirements
echo "Installing/updating package dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Execute the pipeline
echo "Executing the benchmarking pipeline..."
python run_pipeline.py --model mobilenet_v3 --epochs 1 --runs 3

echo "=== Pipeline Completed ==="
echo "Check the 'phase_5_evaluation/results/' folder for metrics and plots."
echo "Check the 'phase_6_deployment/' folder for the exported ONNX model."
