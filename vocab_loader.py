#!/usr/bin/env python3

import torch
import pickle

def safe_load_vocab(vocab_file):
    """
    Safely load vocabulary file with PyTorch compatibility handling
    """
    try:
        # Try loading normally first
        vocab_dump = torch.load(vocab_file, map_location='cpu')
        return vocab_dump
    except Exception as e:
        print(f"Standard loading failed: {e}")
        print("Attempting alternative loading method...")
        
        # Try with pickle protocol handling
        try:
            with open(vocab_file, 'rb') as f:
                vocab_dump = torch.load(f, map_location='cpu', pickle_module=pickle)
            return vocab_dump
        except Exception as e2:
            print(f"Alternative loading also failed: {e2}")
            raise e2 