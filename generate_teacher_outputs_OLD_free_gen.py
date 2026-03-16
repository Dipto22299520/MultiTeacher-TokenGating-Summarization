"""
Offline Teacher Generation Script
==================================
Generates summaries and token-level logits/logprobs from Qwen2.5-32B and Qwen2.5-14B
teachers for the BanSum dataset. Saves outputs to disk for later distillation.

This script is framework-agnostic at the teacher level — you can swap in ANY local
LLM backend (HuggingFace, vLLM, Ollama, llama-cpp-python) by changing the
TeacherBackend class.

Usage:
    python generate_teacher_outputs.py --teacher 32b
    python generate_teacher_outputs.py --teacher 14b
"""

import os
import sys
import json
import math
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *


def load_dataset():
    """Load and split BanSum dataset."""
    print(f"\n{'='*80}")
    print("LOADING BANSUM DATASET")
    print(f"{'='*80}")
    
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total samples loaded: {len(data)}")
    
    # Shuffle with fixed seed for reproducibility
    np.random.seed(SEED)
    indices = np.random.permutation(len(data))
    data = [data[i] for i in indices]
    
    # Split
    total = len(data)
    train_end = int(TRAIN_SPLIT * total)
    val_end = train_end + int(VAL_SPLIT * total)
    
    splits = {
        'train': data[:train_end],
        'validation': data[train_end:val_end],
        'test': data[val_end:]
    }
    
    for name, split_data in splits.items():
        print(f"  {name}: {len(split_data)} samples")
    
    # Limit samples if configured
    if MAX_SAMPLES is not None:
        for name in splits:
            splits[name] = splits[name][:MAX_SAMPLES]
        print(f"\n  [LIMITED TO {MAX_SAMPLES} SAMPLES PER SPLIT]")
    
    return splits


class HuggingFaceTeacherBackend:
    """
    HuggingFace Transformers backend for teacher inference.
    Loads model in 4-bit quantization for memory efficiency.
    
    Can be swapped for vLLM, Ollama, or llama-cpp-python backends.
    """
    
    def __init__(self, model_name, quantization="4bit"):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        
        print(f"\nLoading model: {model_name}")
        print(f"Quantization: {quantization}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            padding_side='left'
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Quantization config
        if quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif quantization == "8bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quant_config = None
        
        load_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
        }
        if quant_config:
            load_kwargs["quantization_config"] = quant_config
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model.eval()
        
        self.vocab_size = self.model.config.vocab_size
        print(f"Model loaded. Vocab size: {self.vocab_size}")
        
        # GPU memory usage
        if torch.cuda.is_available():
            mem_gb = torch.cuda.max_memory_allocated() / 1e9
            print(f"GPU memory used: {mem_gb:.2f} GB")
    
    def generate_with_logits(self, texts, max_new_tokens=256, temperature=0.7, top_p=0.9):
        """
        Generate summaries and extract token-level top-k logprobs.
        
        Returns:
            list of dicts, each containing:
                - 'summary': generated summary text
                - 'token_ids': list of generated token IDs
                - 'top_k_logprobs': list of (token_id, logprob) tuples per position
                - 'full_logprobs': full logprob distribution per position (optional, for KD)
        """
        results = []
        
        for text in texts:
            prompt = TEACHER_PROMPT_TEMPLATE.format(text=text)
            
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=TEACHER_MAX_INPUT_TOKENS
            ).to(self.model.device)
            
            input_len = inputs['input_ids'].shape[1]
            
            with torch.no_grad():
                # Generate with output scores
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            
            # Extract generated tokens (exclude prompt)
            generated_ids = outputs.sequences[0][input_len:].cpu()
            summary = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            # Extract per-token logprobs from scores
            # outputs.scores is a tuple of (num_generated_tokens,) tensors of shape (1, vocab_size)
            token_logprobs_list = []
            token_ids_list = []
            
            for step_idx, score_tensor in enumerate(outputs.scores):
                # score_tensor shape: (1, vocab_size)
                logprobs = F.log_softmax(score_tensor[0].float(), dim=-1)  # (vocab_size,)
                
                # Get top-k
                top_k_vals, top_k_ids = torch.topk(logprobs, k=min(LOGIT_TOP_K, logprobs.shape[0]))
                
                top_k_entries = list(zip(
                    top_k_ids.cpu().tolist(),
                    top_k_vals.cpu().tolist()
                ))
                
                token_logprobs_list.append(top_k_entries)
                token_ids_list.append(generated_ids[step_idx].item())
            
            results.append({
                'summary': summary,
                'token_ids': token_ids_list,
                'top_k_logprobs': token_logprobs_list,
            })
        
        return results


def save_teacher_outputs(results, output_dir, split_name, batch_idx):
    """Save teacher outputs to disk as JSONL for streaming during training."""
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"{split_name}.jsonl")
    mode = 'a' if batch_idx > 0 else 'w'
    
    with open(output_file, mode, encoding='utf-8') as f:
        for result in results:
            # Convert numpy types for JSON serialization
            record = {
                'summary': result['summary'],
                'token_ids': result['token_ids'],
                'top_k_logprobs': result['top_k_logprobs'],
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def generate_for_split(backend, data, output_dir, split_name, batch_size=1):
    """Generate teacher outputs for an entire data split."""
    print(f"\n{'='*80}")
    print(f"GENERATING TEACHER OUTPUTS — {split_name.upper()}")
    print(f"{'='*80}")
    print(f"Samples: {len(data)}, Batch size: {batch_size}")
    
    # Check for existing progress (resume support)
    output_file = os.path.join(output_dir, f"{split_name}.jsonl")
    existing_count = 0
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_count = sum(1 for _ in f)
        print(f"Found {existing_count} existing outputs. Resuming from sample {existing_count}.")
    
    data_remaining = data[existing_count:]
    if len(data_remaining) == 0:
        print("All samples already processed. Skipping.")
        return
    
    num_batches = math.ceil(len(data_remaining) / batch_size)
    
    for batch_idx in tqdm(range(num_batches), desc=f"  {split_name}"):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(data_remaining))
        batch_data = data_remaining[start:end]
        
        # Extract text from batch
        texts = [item[DATASET_TEXT_KEY] for item in batch_data]
        
        # Generate
        try:
            results = backend.generate_with_logits(
                texts,
                max_new_tokens=TEACHER_MAX_OUTPUT_TOKENS,
                temperature=TEACHER_TEMPERATURE,
                top_p=TEACHER_TOP_P
            )
            
            # Add original data IDs for traceability
            for i, result in enumerate(results):
                result['original_id'] = batch_data[i].get('ID', existing_count + start + i)
                result['gold_summary'] = batch_data[i][DATASET_SUMMARY_KEY]
            
            # Save incrementally
            save_teacher_outputs(results, output_dir, split_name, existing_count + batch_idx)
            
        except Exception as e:
            print(f"\n  ERROR at batch {batch_idx}: {e}")
            print(f"  Skipping batch. Progress saved up to batch {batch_idx}.")
            continue
    
    # Verify count
    with open(output_file, 'r', encoding='utf-8') as f:
        final_count = sum(1 for _ in f)
    print(f"\n  Completed: {final_count}/{len(data)} samples saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate teacher outputs for distillation")
    parser.add_argument(
        "--teacher", 
        type=str, 
        required=True, 
        choices=["32b", "14b"],
        help="Which teacher to run: 32b or 14b"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "validation", "test", "all"],
        help="Which data split to generate for"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for generation (1 recommended for large models)"
    )
    args = parser.parse_args()
    
    # Select model and output directory
    if args.teacher == "32b":
        model_name = TEACHER_32B_MODEL
        output_dir = TEACHER_32B_OUTPUTS
    else:
        model_name = TEACHER_14B_MODEL
        output_dir = TEACHER_14B_OUTPUTS
    
    print(f"\n{'='*80}")
    print(f"DUAL-TEACHER KNOWLEDGE DISTILLATION — TEACHER GENERATION")
    print(f"{'='*80}")
    print(f"Teacher: {model_name}")
    print(f"Output: {output_dir}")
    print(f"Dataset: {DATASET_FILE}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load dataset
    splits = load_dataset()
    
    # Load teacher model
    backend = HuggingFaceTeacherBackend(model_name, quantization=TEACHER_QUANTIZATION)
    
    # Generate for requested splits
    splits_to_run = list(splits.keys()) if args.split == "all" else [args.split]
    
    for split_name in splits_to_run:
        generate_for_split(
            backend=backend,
            data=splits[split_name],
            output_dir=output_dir,
            split_name=split_name,
            batch_size=args.batch_size,
        )
    
    print(f"\n{'='*80}")
    print("TEACHER GENERATION COMPLETE!")
    print(f"{'='*80}")
    print(f"Outputs saved to: {output_dir}")
    
    # Save metadata
    metadata = {
        "teacher_model": model_name,
        "quantization": TEACHER_QUANTIZATION,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "dataset": DATASET_FILE,
        "logit_top_k": LOGIT_TOP_K,
        "max_output_tokens": TEACHER_MAX_OUTPUT_TOKENS,
        "temperature": TEACHER_TEMPERATURE,
        "top_p": TEACHER_TOP_P,
        "splits_generated": splits_to_run,
        "samples_per_split": {name: len(splits[name]) for name in splits_to_run},
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
