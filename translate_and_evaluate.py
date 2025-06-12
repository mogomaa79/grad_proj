#!/usr/bin/env python3

import os
import sys
import subprocess

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"{description}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error in {description}: {result.stderr}")
        return False
    print(result.stdout)
    return True

def main():
    print("Starting translation and evaluation...")
    
    # Configuration
    num_steps_to_run_kd = 100000
    model_path = f"output/kd-model/ckpt/model_step_{num_steps_to_run_kd}.pt"
    data_dir = "data/de-en"
    src_file = f"{data_dir}/test.de.bert"
    tgt_file = f"{data_dir}/test.en.bert"
    out_dir = "output/translation"
    ref_file = f"{data_dir}/test.en"
    
    # Ensure the output directory exists
    os.makedirs(out_dir, exist_ok=True)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        print("You need to train the model first or adjust the model path to point to an existing checkpoint.")
        sys.exit(1)
    
    print(f"Model found at {model_path}. Running translation...")
    
    # Run translation
    translate_cmd = f"""python opennmt/translate.py -model {model_path} \
                       -src {src_file} \
                       -tgt {tgt_file} \
                       -output {out_dir}/result.en \
                       -gpu 0 \
                       -beam_size 5 -alpha 0.6 \
                       -length_penalty wu"""
    
    if not run_command(translate_cmd, "Translation"):
        sys.exit(1)
    
    # Check if translation output exists
    if not os.path.exists(f"{out_dir}/result.en"):
        print("Warning: Translation output file was not generated.")
        sys.exit(1)
    
    print("Translation completed. Detokenizing output...")
    
    # Detokenize output
    detokenize_cmd = f"python scripts/bert_detokenize.py --file {out_dir}/result.en --output_dir {out_dir}"
    
    if not run_command(detokenize_cmd, "Detokenization"):
        sys.exit(1)
    
    # Check if detokenized output exists
    if not os.path.exists(f"{out_dir}/result.en.detok"):
        print("Warning: Detokenized output file was not generated.")
        sys.exit(1)
    
    print("Evaluating with BLEU score...")
    
    # Evaluate with BLEU
    bleu_cmd = f"perl opennmt/tools/multi-bleu.perl {ref_file} < {out_dir}/result.en.detok > {out_dir}/result.bleu"
    
    if not run_command(bleu_cmd, "BLEU evaluation"):
        print("Warning: BLEU evaluation failed, but continuing...")
    
    # Display BLEU score if file exists
    if os.path.exists(f"{out_dir}/result.bleu"):
        with open(f"{out_dir}/result.bleu", "r") as f:
            bleu_score = f.read().strip()
            print(f"BLEU Score: {bleu_score}")
    else:
        print("Warning: BLEU score file was not generated.")
    
    print("Translation and evaluation completed!")

if __name__ == "__main__":
    main() 