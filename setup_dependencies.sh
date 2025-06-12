#!/bin/bash

echo "Setting up dependencies for language distillation..."

# Clone the repository
if [ ! -d "language_distilling" ]; then
    git clone https://github.com/ziadtarek12/language_distilling
fi

cd language_distilling
git checkout eval

# Uninstall existing torch packages to avoid conflicts
pip uninstall -y torch torchvision torchaudio

# Install required packages
pip install transformers==4.26.0
pip install pytorch-pretrained-bert
pip install cytoolz
pip install tqdm
pip install torchtext==0.16.0
pip install torchvision==0.16.0
pip install torch==2.1.0
pip install torchaudio==2.1.0
pip install configargparse
pip install tensorboardX
pip install ipdb
pip install matplotlib

echo "Dependencies installed successfully!" 