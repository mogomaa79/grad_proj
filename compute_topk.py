#!/usr/bin/env python3

import os
import sys
import torch
import shelve
from tqdm import tqdm

# Add paths
sys.path.append('.')
sys.path.append('./opennmt')

# Import functions for top-k computation
from dump_teacher_topk import tensor_loads, dump_topk

def main():
    print("Starting top-k computation...")
    
    # Check device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Configuration
    bert_dump_path = "output/bert_dump"
    k = 8  # Following the paper
    
    # Check if required files exist
    linear_path = f'{bert_dump_path}/linear.pt'
    db_path = f'{bert_dump_path}/db'
    
    if not os.path.exists(linear_path):
        print(f"Error: Linear layer not found at {linear_path}")
        print("Please run hidden states extraction first.")
        sys.exit(1)
    
    # Check if hidden states database exists
    if not any(os.path.exists(f"{db_path}{ext}") for ext in ["", ".db", ".dat", ".bak", ".dir"]):
        print(f"Error: Hidden states database not found at {db_path}")
        print("Please run hidden states extraction first.")
        sys.exit(1)
    
    # Load linear layer
    linear = torch.load(linear_path, map_location='cpu')
    # Ensure the linear layer uses the same precision as the hidden states (FP16/Half)
    linear = linear.half()
    linear.to(device)
    
    # Compute top-k logits
    print("Computing top-k logits...")
    with shelve.open(f'{bert_dump_path}/db', 'r') as db, \
         shelve.open(f'{bert_dump_path}/topk', 'c') as topk_db:
        for key, value in tqdm(db.items(), total=len(db), desc='Computing topk...'):
            bert_hidden = torch.tensor(tensor_loads(value)).to(device)
            # bert_hidden is already in half precision, no need to convert
            topk = linear(bert_hidden).topk(dim=-1, k=k)
            dump = dump_topk(topk)
            topk_db[key] = dump
            
            # Clear tensor from GPU memory after each iteration
            del bert_hidden
            torch.cuda.empty_cache()
    
    # Final memory cleanup
    print("Clearing GPU memory...")
    linear.cpu()  # Move linear layer to CPU
    del linear     # Delete the linear layer
    torch.cuda.empty_cache()  # Empty the CUDA cache
    print("GPU memory cleared after top-k computation")
    print(f"Top-k logits computed and saved to {bert_dump_path}/topk")

if __name__ == "__main__":
    main() 