#!/usr/bin/env python3

import os
import sys
import torch
import numpy as np
import random
from transformers import BertTokenizer

# Add paths
sys.path.append('.')
sys.path.append('./opennmt')

# Import tokenization functions
from scripts.bert_tokenize import tokenize, process

def setup_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    print("Starting BERT tokenization...")
    
    # Set seed for reproducibility
    setup_seed(42)
    
    # Check device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load BERT tokenizer
    bert_model = "bert-base-multilingual-cased"
    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case='uncased' in bert_model)
    
    # Define data directories
    data_dir = "data/de-en"
    
    # BERT tokenize our dataset files
    for language in ['de', 'en']:
        for split in ['train', 'valid', 'test']:
            input_file = f"{data_dir}/{split}.{language}"
            output_file = f"{data_dir}/{split}.{language}.bert"
            print(f"Tokenizing {input_file}...")
            
            with open(input_file, 'r') as reader, open(output_file, 'w') as writer:
                process(reader, writer, tokenizer)
    
    print("BERT tokenization completed!")

if __name__ == "__main__":
    main() 