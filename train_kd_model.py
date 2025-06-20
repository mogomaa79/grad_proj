#!/usr/bin/env python3

import os
import sys
import torch
import yaml
import argparse
from torch.utils.data import DataLoader

# Add paths
sys.path.append('.')
sys.path.append('./opennmt')

# Import required modules for training
from onmt.inputters.bert_kd_dataset import BertKdDataset, TokenBucketSampler
from onmt.utils.optimizers import Optimizer
from onmt.train_single import build_model_saver, build_trainer, cycle_loader
from onmt.model_builder import build_model

def setup_args():
    """Setup training arguments"""
    config_path = "opennmt/config/config-transformer-base-mt-deen.yml"
    
    # Load configuration
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    
    # Create args object
    args = argparse.Namespace(**config)
    
    # Setup KD parameters
    args.train_from = None
    args.max_grad_norm = None
    args.kd_topk = 8
    args.train_steps = 1
    args.kd_temperature = 10.0
    args.kd_alpha = 0.5
    args.warmup_steps = 1
    args.learning_rate = 2.0
    args.bert_dump = "output/bert_dump"
    args.data_db = "data/DEEN.db"
    args.bert_kd = True
    args.data = "data/DEEN"
    
    # Add missing required parameters
    args.model_type = "text"
    args.copy_attn = False
    args.global_attention = "general"
    
    # Add embeddings parameters
    args.src_word_vec_size = args.word_vec_size
    args.tgt_word_vec_size = args.word_vec_size
    args.feat_merge = "concat"
    args.feat_vec_size = -1
    args.feat_vec_exponent = 0.7
    
    # Add pretrained word vectors parameters
    args.pre_word_vecs_enc = None
    args.pre_word_vecs_dec = None
    args.pre_word_vecs = None
    args.fix_word_vecs_enc = False
    args.fix_word_vecs_dec = False
    
    # Add critical RNN and transformer parameters
    args.enc_rnn_size = args.rnn_size
    args.dec_rnn_size = args.rnn_size
    args.transformer_ff = getattr(args, 'transformer_ff', 2048)
    args.heads = getattr(args, 'heads', 8)
    
    # Add transformer position parameters
    args.max_relative_positions = 0
    args.position_encoding = True
    args.param_init = 0.0
    args.param_init_glorot = True
    
    # Fix share_embeddings
    args.share_embeddings = False
    args.share_decoder_embeddings = False
    
    # Add training parameters needed by OpenNMT trainer
    args.truncated_decoder = 0
    args.max_generator_batches = getattr(args, 'max_generator_batches', 32)
    args.normalization = getattr(args, 'normalization', 'sents')
    args.accum_count = getattr(args, 'accum_count', 1)
    args.accum_steps = [0]
    args.average_decay = 0.0
    args.average_every = 1
    args.report_manager = None
    args.valid_steps = getattr(args, 'valid_steps', 1)
    args.early_stopping = 0
    args.early_stopping_criteria = None
    args.valid_batch_size = 32
    
    # Add the missing transformer attention parameters
    args.self_attn_type = "scaled-dot"
    args.input_feed = 1
    args.copy_attn_type = None
    args.generator_function = "softmax"
    
    # Add distributed training parameters
    args.local_rank = -1
    args.gpu_ranks = getattr(args, 'gpu_ranks', [0])
    args.gpu_verbose_level = 0
    args.world_size = getattr(args, 'world_size', 1)
    
    # Add other required parameters
    args.encoder_type = getattr(args, 'encoder_type', "transformer")
    args.decoder_type = getattr(args, 'decoder_type', "transformer") 
    args.enc_layers = getattr(args, 'layers', 6)
    args.dec_layers = getattr(args, 'layers', 6)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'dropout', 0.1)
    args.bridge = ""
    args.aux_tune = False
    args.subword_prefix = "▁"
    args.subword_prefix_is_joiner = False
    
    # Output paths
    output_path = "output/kd-model"
    args.save_model = os.path.join(output_path, 'ckpt', 'model')
    args.log_file = os.path.join(output_path, 'log', 'log')
    args.tensorboard_log_dir = os.path.join(output_path, 'log')
    
    return args

def check_prerequisites():
    """Check if all required files exist"""
    required_files = [
        "data/DEEN.db",
        "data/DEEN.vocab.pt",
        "output/bert_dump/topk"
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            # For shelve databases, check for various extensions
            if "topk" in file_path:
                if not any(os.path.exists(f"{file_path}{ext}") for ext in ["", ".db", ".dat", ".bak", ".dir"]):
                    print(f"Error: Required file not found: {file_path}")
                    return False
            else:
                print(f"Error: Required file not found: {file_path}")
                return False
    
    return True

def manual_train_iter(train_iter, train_loader, device):
    """Custom iterator that provides batches without step limitation"""
    while True:
        try:
            batch = next(train_iter)
        except StopIteration:
            print("Restarting data iterator")
            train_iter = cycle_loader(train_loader, device)
            batch = next(train_iter)
        yield batch

def main():
    print("Starting knowledge distillation training...")
    
    # Check prerequisites
    if not check_prerequisites():
        print("Please run the previous stages first.")
        sys.exit(1)
    
    # Check device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Setup arguments
    args = setup_args()
    
    # Load vocabulary and dataset
    vocab = torch.load(args.data + '.vocab.pt')
    src_vocab = vocab['src'].fields[0][1].vocab.stoi
    tgt_vocab = vocab['tgt'].fields[0][1].vocab.stoi
    
    # Create dataset
    train_dataset = BertKdDataset(args.data_db, args.bert_dump, 
                                 src_vocab, tgt_vocab,
                                 max_len=150, k=args.kd_topk)
    
    # Create data loader
    BUCKET_SIZE = 8192
    train_sampler = TokenBucketSampler(
        train_dataset.keys, BUCKET_SIZE, 6144,
        batch_multiple=1)
    
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                             num_workers=4,
                             collate_fn=BertKdDataset.pad_collate)
    
    train_iter = cycle_loader(train_loader, device)
    
    # Build the model
    model = build_model(args, args, fields=vocab, checkpoint=None)
    model.to(device)
    
    # Build optimizer
    optim = Optimizer.from_opt(model, args, checkpoint=None)
    
    # Build model saver
    model_saver = build_model_saver(args, args, model, vocab, optim)
    
    # Build trainer
    trainer = build_trainer(args, 0, model, vocab, optim, model_saver=model_saver)
    
    # Training parameters
    num_steps_to_run_kd = 1  # Full training steps
    
    # Make sure the optimizer is tracking the step correctly
    if not hasattr(optim, 'training_step'):
        optim._training_step = 0
    
    print("Starting model training with knowledge distillation...")
    # Train the model
    trainer.train(
        manual_train_iter(train_iter, train_loader, device),
        num_steps_to_run_kd,
        save_checkpoint_steps=1,  # Save every step for debugging
        valid_iter=None
    )
    
    print(f"Model trained for {num_steps_to_run_kd} steps and saved to output/kd-model/ckpt")

if __name__ == "__main__":
    main() 