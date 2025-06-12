# Language Distillation Pipeline

This repository contains a complete implementation of the language distillation pipeline for neural machine translation, based on the research paper methodology. The pipeline implements knowledge distillation from a fine-tuned BERT model to a smaller Transformer model for German-English translation.

## Overview

The pipeline consists of 9 sequential stages:

1. **Setup Dependencies** - Install required packages and clone the repository
2. **Download Data** - Download and prepare the IWSLT German-English dataset
3. **BERT Tokenization** - Tokenize the dataset using BERT tokenizer
4. **Data Preprocessing** - Create vocabulary and preprocess data for training
5. **CMLM Fine-tuning** - Fine-tune BERT with Conditional Masked Language Modeling
6. **Hidden States Extraction** - Extract hidden states from the fine-tuned BERT model
7. **Top-K Computation** - Compute top-k logits from hidden states
8. **Knowledge Distillation Training** - Train student model with knowledge distillation
9. **Translation and Evaluation** - Run translation and compute BLEU scores

## Requirements

### System Requirements
- Python 3.7+
- CUDA-capable GPU (recommended for training)
- At least 16GB RAM
- At least 50GB free disk space

### Software Dependencies
- Python packages (automatically installed):
  - torch==2.1.0
  - transformers==4.26.0
  - torchtext==0.16.0
  - tqdm, tensorboardX, configargparse, etc.
- System tools:
  - git
  - perl (for BLEU evaluation)

## Quick Start

### Option 1: Run Complete Pipeline (Recommended)

```bash
# Make the master script executable
chmod +x run_language_distillation.sh

# Run the complete pipeline
./run_language_distillation.sh
```

This will run all 9 stages sequentially. The process will take several hours, especially the training stages (5 and 8).

### Option 2: Run Individual Stages

You can also run individual stages manually:

```bash
# Stage 1: Setup
chmod +x setup_dependencies.sh
./setup_dependencies.sh
cd language_distilling

# Stage 2: Download data
chmod +x ../download_data.sh
../download_data.sh

# Stage 3: BERT tokenization
python ../bert_tokenize_data.py

# Stage 4: Preprocessing
python ../preprocess_data.py

# Stage 5: CMLM fine-tuning
python ../train_cmlm.py

# Stage 6: Extract hidden states
python ../extract_hidden_states.py

# Stage 7: Compute top-k
python ../compute_topk.py

# Stage 8: Knowledge distillation training
python ../train_kd_model.py

# Stage 9: Translation and evaluation
python ../translate_and_evaluate.py
```

## File Structure

After running the pipeline, the following structure will be created:

```
language_distilling/
├── data/
│   ├── de-en/                    # Raw and tokenized datasets
│   ├── DEEN.db                   # Preprocessed training data
│   └── DEEN.vocab.pt             # Vocabulary file
├── output/
│   ├── cmlm_model/               # Fine-tuned BERT model
│   ├── bert_dump/                # Hidden states and top-k logits
│   ├── kd-model/                 # Knowledge distillation model
│   └── translation/              # Translation results and BLEU scores
└── scripts/                      # Utility scripts
```

## Scripts Description

### Core Scripts

- **`run_language_distillation.sh`** - Master script that runs the complete pipeline
- **`setup_dependencies.sh`** - Installs dependencies and clones repository
- **`download_data.sh`** - Downloads and prepares IWSLT dataset

### Python Scripts

- **`vocab_loader.py`** - Utility for loading vocabulary with PyTorch compatibility
- **`bert_tokenize_data.py`** - BERT tokenization of the dataset
- **`preprocess_data.py`** - Data preprocessing and vocabulary creation
- **`train_cmlm.py`** - CMLM fine-tuning of BERT model
- **`extract_hidden_states.py`** - Hidden states extraction from fine-tuned BERT
- **`compute_topk.py`** - Top-k logits computation
- **`train_kd_model.py`** - Knowledge distillation training
- **`translate_and_evaluate.py`** - Translation and BLEU evaluation

## Configuration

### Training Parameters

The pipeline uses the following key parameters (can be modified in the respective scripts):

- **CMLM Training**: 100,000 steps, learning rate 5e-5
- **Knowledge Distillation**: 100,000 steps, temperature 10.0, alpha 0.5
- **Top-k**: k=8 (following the paper)
- **Batch sizes**: Dynamically determined by token bucket sampling

### Hardware Considerations

- **GPU Memory**: The pipeline requires significant GPU memory (8GB+ recommended)
- **Training Time**: 
  - CMLM fine-tuning: ~2-4 hours on modern GPU
  - Knowledge distillation: ~3-5 hours on modern GPU
  - Other stages: <1 hour each

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in the training scripts
   - Enable gradient checkpointing
   - Use a GPU with more memory

2. **Missing Dependencies**
   - Ensure all system dependencies are installed
   - Check Python version compatibility

3. **Data Download Issues**
   - Check internet connection
   - Manually download IWSLT dataset if needed

4. **PyTorch Compatibility**
   - The `vocab_loader.py` handles most compatibility issues
   - Ensure consistent PyTorch versions

### Debug Mode

For faster debugging, you can enable debug mode in `extract_hidden_states.py`:
```python
debug_mode = True  # Process only 100 samples instead of full dataset
```

## Expected Results

The pipeline should achieve:
- BLEU scores comparable to the original paper
- Successful knowledge transfer from BERT to the student model
- Significant speedup in inference compared to the teacher model

## Citation

If you use this pipeline in your research, please cite the original paper and this implementation.

## License

This project follows the same license as the original language_distilling repository.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the individual script outputs for error messages
3. Ensure all prerequisites are met 