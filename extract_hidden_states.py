#!/usr/bin/env python3

import os
import sys
import torch
import shelve
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm

# Add paths
sys.path.append('.')
sys.path.append('./opennmt')

# Import extraction functions
from dump_teacher_hiddens import tensor_dumps, gather_hiddens, BertSampleDataset, batch_features, process_batch
from cmlm.model import BertForSeq2seq

def build_db_batched(corpus_path, out_db, bert, toker, batch_size=8, debug_mode=False, max_samples=100):
    """Extract hidden states with optional debugging mode"""
    dataset = BertSampleDataset(corpus_path, toker)
    
    # For debugging, limit the number of samples
    if debug_mode:
        all_ids = dataset.ids
        subset_ids = all_ids[:max_samples] if len(all_ids) > max_samples else all_ids
        dataset.ids = subset_ids
        print(f"DEBUG MODE: Processing only {len(subset_ids)} samples instead of {len(all_ids)}")
    
    loader = DataLoader(dataset, batch_size=batch_size,
                       num_workers=4, collate_fn=batch_features)
    
    with tqdm(desc='Computing BERT features', total=len(dataset)) as pbar:
        for ids, *batch in loader:
            outputs = process_batch(batch, bert, toker)
            for id_, output in zip(ids, outputs):
                if output is not None:
                    out_db[id_] = tensor_dumps(output)
            pbar.update(len(ids))
            
            # For debugging, break after the first batch if needed
            if debug_mode and batch_size >= max_samples:
                print("First batch processed, breaking early due to debug mode")
                break

def main():
    print("Starting hidden states extraction...")
    
    # Check device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Configuration
    bert_model = "bert-base-multilingual-cased"
    output_dir = "output/cmlm_model"
    bert_dump_path = "output/bert_dump"
    num_steps_to_run = 100000  # Should match the training steps
    
    # Path to model checkpoint from Stage 1
    ckpt_path = f"{output_dir}/model_step_{num_steps_to_run}.pt"
    
    if not os.path.exists(ckpt_path):
        print(f"Error: Model checkpoint not found at {ckpt_path}")
        print("Please run CMLM training first.")
        sys.exit(1)
    
    # Load BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case='uncased' in bert_model)
    
    # Load the fine-tuned BERT model
    state_dict = torch.load(ckpt_path, map_location='cpu')
    vsize = state_dict['cls.predictions.decoder.weight'].size(0)
    bert = BertForSeq2seq.from_pretrained(bert_model).eval()
    bert.to(device)
    
    # Resize the model layers to match the exact dimensions from the checkpoint
    print(f"Resizing model to exact vocabulary size: {vsize}")
    hidden_size = bert.config.hidden_size
    
    # Create exact-sized layers without padding to multiples of 8
    bert.cls.predictions.decoder = torch.nn.Linear(hidden_size, vsize, bias=True)
    bert.cls.predictions.bias = bert.cls.predictions.decoder.bias
    bert.config.vocab_size = vsize
    
    # Now load the state dict - should have matching dimensions
    bert.load_state_dict(state_dict)
    
    # Save the final projection layer
    linear = torch.nn.Linear(bert.config.hidden_size, bert.config.vocab_size)
    linear.weight.data = state_dict['cls.predictions.decoder.weight']
    linear.bias.data = state_dict['cls.predictions.bias']
    torch.save(linear, f'{bert_dump_path}/linear.pt')
    
    # Extract hidden states
    db_path = "data/DEEN.db"
    print("Extracting hidden states...")
    
    # Set debug mode to False for full processing, True for quick debugging
    debug_mode = False  # Change to True for faster debugging
    max_samples = 100   # Number of samples to process in debug mode
    
    with shelve.open(f'{bert_dump_path}/db', 'c') as out_db, torch.no_grad():
        build_db_batched(db_path, out_db, bert, tokenizer, batch_size=8, 
                        debug_mode=debug_mode, max_samples=max_samples)
    
    # Free up GPU memory after extraction
    print("Clearing GPU memory...")
    bert.cpu()  # Move model to CPU
    del bert    # Delete the model
    linear.cpu()  # Move linear layer to CPU
    torch.cuda.empty_cache()  # Empty the CUDA cache
    print("GPU memory cleared after hidden states extraction")
    
    if debug_mode:
        print(f"DEBUG MODE: Hidden states for {max_samples} samples extracted to {bert_dump_path}/db")
        print("To run full extraction, set debug_mode=False")
    else:
        print(f"Hidden states extracted and saved to {bert_dump_path}/db")

if __name__ == "__main__":
    main() 