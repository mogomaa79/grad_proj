#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess

# Add paths
sys.path.append('.')
sys.path.append('./opennmt')

# Import preprocessing functions
from scripts.bert_prepro import main as bert_prepro

def main():
    print("Starting data preprocessing...")
    
    data_dir = "data/de-en"
    
    # Create dataset DB for BERT training
    print("Creating BERT preprocessing database...")
    prepro_args = argparse.Namespace(
        src=f"{data_dir}/train.de.bert",
        tgt=f"{data_dir}/train.en.bert",
        output='data/DEEN'
    )
    
    # Run preprocessing
    bert_prepro(prepro_args)
    
    # Create vocabulary file using OpenNMT's preprocess.py
    print("Creating vocabulary files with OpenNMT preprocess.py...")
    cmd = [
        'python', 'opennmt/preprocess.py',
        '-train_src', f'{data_dir}/train.de.bert',
        '-train_tgt', f'{data_dir}/train.en.bert',
        '-valid_src', f'{data_dir}/valid.de.bert',
        '-valid_tgt', f'{data_dir}/valid.en.bert',
        '-save_data', 'data/DEEN',
        '-src_seq_length', '150',
        '-tgt_seq_length', '150'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running OpenNMT preprocess: {result.stderr}")
        sys.exit(1)
    
    print("Data preprocessing completed!")
    print("Vocabulary file created: data/DEEN.vocab.pt")

if __name__ == "__main__":
    main() 