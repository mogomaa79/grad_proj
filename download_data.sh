#!/bin/bash

echo "Setting up data directories and downloading dataset..."

# Create directories for data and outputs
mkdir -p data/
mkdir -p output/cmlm_model
mkdir -p output/bert_dump
mkdir -p output/kd-model/ckpt
mkdir -p output/kd-model/log
mkdir -p output/translation

# Download IWSLT German-English dataset using the provided script
bash scripts/download-iwslt_deen.sh

echo "Data directories created and dataset downloaded!" 