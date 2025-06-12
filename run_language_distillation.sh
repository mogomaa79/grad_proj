#!/bin/bash

# Language Distillation Pipeline - Master Script
# This script runs the complete language distillation pipeline sequentially

set -e  # Exit on any error

echo "=========================================="
echo "Language Distillation Pipeline"
echo "=========================================="

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print stage headers
print_stage() {
    echo ""
    echo "=========================================="
    echo "STAGE $1: $2"
    echo "=========================================="
}

# Check prerequisites
echo "Checking prerequisites..."
if ! command_exists python; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

if ! command_exists pip; then
    echo "Error: pip is not installed or not in PATH"
    exit 1
fi

if ! command_exists git; then
    echo "Error: git is not installed or not in PATH"
    exit 1
fi

if ! command_exists perl; then
    echo "Warning: perl is not installed. BLEU evaluation may not work."
fi

echo "Prerequisites check completed."

# Stage 1: Setup Dependencies
print_stage "1" "Setting up dependencies and cloning repository"
chmod +x setup_dependencies.sh
./setup_dependencies.sh

# Change to the cloned repository directory
cd language_distilling

# Copy our scripts to the repository directory
cp ../vocab_loader.py .
cp ../bert_tokenize_data.py .
cp ../preprocess_data.py .
cp ../train_cmlm.py .
cp ../extract_hidden_states.py .
cp ../compute_topk.py .
cp ../train_kd_model.py .
cp ../translate_and_evaluate.py .

# Stage 2: Download Data
print_stage "2" "Downloading and preparing data"
chmod +x ../download_data.sh
../download_data.sh

# Stage 3: BERT Tokenization
print_stage "3" "BERT tokenization of dataset"
python bert_tokenize_data.py

# Stage 4: Data Preprocessing
print_stage "4" "Data preprocessing and vocabulary creation"
python preprocess_data.py

# Stage 5: CMLM Fine-tuning
print_stage "5" "CMLM (Conditional Masked Language Model) fine-tuning"
echo "This stage will take a significant amount of time (several hours with GPU)..."
python train_cmlm.py

# Stage 6: Hidden States Extraction
print_stage "6" "Extracting hidden states from fine-tuned BERT model"
python extract_hidden_states.py

# Stage 7: Top-K Computation
print_stage "7" "Computing top-k logits from hidden states"
python compute_topk.py

# Stage 8: Knowledge Distillation Training
print_stage "8" "Training student model with knowledge distillation"
echo "This stage will take a significant amount of time (several hours with GPU)..."
python train_kd_model.py

# Stage 9: Translation and Evaluation
print_stage "9" "Running translation and BLEU evaluation"
python translate_and_evaluate.py

# Final summary
echo ""
echo "=========================================="
echo "PIPELINE COMPLETED SUCCESSFULLY!"
echo "=========================================="
echo ""
echo "Results can be found in:"
echo "  - CMLM model: output/cmlm_model/"
echo "  - KD model: output/kd-model/ckpt/"
echo "  - Translation results: output/translation/"
echo "  - BERT dumps: output/bert_dump/"
echo ""
echo "Check the translation results and BLEU score in output/translation/"
echo ""

# Return to original directory
cd ..

echo "Language distillation pipeline completed!" 