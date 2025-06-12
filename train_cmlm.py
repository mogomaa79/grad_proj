#!/usr/bin/env python3

import os
import sys
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup

# Add paths
sys.path.append('.')
sys.path.append('./opennmt')

# Import needed modules
from cmlm.data import BertDataset, TokenBucketSampler
from cmlm.model import convert_embedding, BertForSeq2seq
from cmlm.util import Logger, RunningMeter
from vocab_loader import safe_load_vocab

def setup_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    print("Starting CMLM fine-tuning...")
    
    # Set seed for reproducibility
    setup_seed(42)
    
    # Check device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Configuration
    bert_model = "bert-base-multilingual-cased"
    vocab_file = "data/DEEN.vocab.pt"
    train_file = "data/DEEN.db"
    data_dir = "data/de-en"
    valid_src = f"{data_dir}/valid.de.bert"
    valid_tgt = f"{data_dir}/valid.en.bert"
    output_dir = "output/cmlm_model"
    
    # Load BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case='uncased' in bert_model)
    
    # Load vocabulary using custom loader to avoid PyTorch compatibility issues
    vocab_dump = safe_load_vocab(vocab_file)
    vocab = vocab_dump['tgt'].fields[0][1].vocab.stoi
    
    # Create dataset
    train_dataset = BertDataset(train_file, tokenizer, vocab, seq_len=512, max_len=150)
    
    # Define sampler and data loader
    BUCKET_SIZE = 8192
    train_sampler = TokenBucketSampler(
        train_dataset.lens, BUCKET_SIZE, 6144, batch_multiple=1)
    
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                             num_workers=4,
                             collate_fn=BertDataset.pad_collate)
    
    # Prepare model
    model = BertForSeq2seq.from_pretrained(bert_model)
    bert_embedding = model.bert.embeddings.word_embeddings.weight
    
    # Print model information before modifications
    hidden_size = model.config.hidden_size
    print(f"Original model: BERT hidden size = {hidden_size}")
    print(f"Original model: BERT vocab size = {bert_embedding.size(0)}")
    print(f"Target vocabulary size = {len(vocab)}")
    
    # Convert vocabulary to embedding form
    embedding = convert_embedding(tokenizer, vocab, bert_embedding)
    
    # Update model architecture to accommodate the new vocabulary size
    print(f"Updating model architecture for vocabulary size: {embedding.size(0)}")
    # Create a new decoder with correct dimensions
    model.cls.predictions.decoder = torch.nn.Linear(hidden_size, embedding.size(0), bias=True)
    model.cls.predictions.bias = torch.nn.Parameter(torch.zeros(embedding.size(0)))
    model.config.vocab_size = embedding.size(0)
    
    # Update the weights
    model.cls.predictions.decoder.weight.data.copy_(embedding.data)
    
    # Move model to device
    model.to(device)
    print(f"Model adapted with vocabulary size: {model.config.vocab_size}")
    
    # Training parameters
    learning_rate = 5e-5
    warmup_proportion = 0.1
    max_steps = 100000
    num_steps_to_run = 100000  # Full training steps
    
    # Optimizer using modern AdamW from transformers
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(max_steps * warmup_proportion),
        num_training_steps=max_steps
    )
    
    # Training loop
    running_loss = RunningMeter('loss')
    model.train()
    
    print("Starting CMLM fine-tuning...")
    train_iter = iter(train_loader)
    for step in range(num_steps_to_run):
        try:
            batch = next(train_iter)
        except StopIteration:
            # Restart iterator if we run out of batches
            train_iter = iter(train_loader)
            batch = next(train_iter)
            
        # Move batch to device
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, lm_label_ids = batch
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Create output mask from lm_label_ids for model forward pass
        output_mask = lm_label_ids != -1  # Masking for non-padded tokens
        
        # Forward pass with output_mask parameter
        loss = model(input_ids, segment_ids, input_mask, lm_label_ids, output_mask)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        running_loss(loss.item())
        
        if step % 100 == 0:
            print(f"Step {step}, Loss: {running_loss.val:.4f}")
            # Clear CUDA cache periodically to avoid memory issues
            torch.cuda.empty_cache()
    
    # Save model checkpoint
    torch.save(model.state_dict(), f"{output_dir}/model_step_{num_steps_to_run}.pt")
    print(f"Model saved to {output_dir}/model_step_{num_steps_to_run}.pt")
    print("CMLM fine-tuning completed!")

if __name__ == "__main__":
    main() 