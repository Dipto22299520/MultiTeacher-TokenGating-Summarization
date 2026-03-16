"""
Student Distillation Training Script
======================================
Trains Qwen2.5-3B student with LoRA using pre-computed teacher logits.
Supports all 8 experiment configurations (3 baselines + 5 ablations).

Usage:
    python train_student.py --experiment ewad_full
    python train_student.py --experiment baseline_no_distill
    python train_student.py --experiment ewad_cpdp
    
For all experiments:
    python run_all_experiments.py
"""

import os
import sys
import json
import math
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from ewad_cpdp_loss import (
    DualTeacherDistillationLoss,
    batch_sparse_to_dense,
    sparse_topk_to_dense
)


# ============================================================================
# Dataset
# ============================================================================
class DistillationDataset(Dataset):
    """
    Dataset that combines gold labels with pre-computed teacher outputs.
    Loads teacher outputs from JSONL files lazily for memory efficiency.
    """
    
    def __init__(
        self, 
        data_samples,          # list of dicts from BanSum
        teacher_32b_file,      # path to teacher_32b JSONL
        teacher_14b_file,      # path to teacher_14b JSONL
        tokenizer,
        max_input_tokens,
        max_output_tokens,
        use_distillation=True,
    ):
        self.data_samples = data_samples
        self.tokenizer = tokenizer
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.use_distillation = use_distillation
        
        # Load teacher outputs into memory (they're JSONL)
        self.teacher_32b_outputs = None
        self.teacher_14b_outputs = None
        
        if use_distillation and teacher_32b_file and os.path.exists(teacher_32b_file):
            print(f"  Loading 32B teacher outputs from {teacher_32b_file}...")
            self.teacher_32b_outputs = self._load_jsonl(teacher_32b_file)
            print(f"  Loaded {len(self.teacher_32b_outputs)} teacher 32B outputs")
        
        if use_distillation and teacher_14b_file and os.path.exists(teacher_14b_file):
            print(f"  Loading 14B teacher outputs from {teacher_14b_file}...")
            self.teacher_14b_outputs = self._load_jsonl(teacher_14b_file)
            print(f"  Loaded {len(self.teacher_14b_outputs)} teacher 14B outputs")
        
        # Validate alignment
        if self.teacher_32b_outputs and len(self.teacher_32b_outputs) != len(self.data_samples):
            print(f"  WARNING: 32B outputs ({len(self.teacher_32b_outputs)}) != data samples ({len(self.data_samples)})")
            min_len = min(len(self.teacher_32b_outputs), len(self.data_samples))
            self.data_samples = self.data_samples[:min_len]
            self.teacher_32b_outputs = self.teacher_32b_outputs[:min_len]
            if self.teacher_14b_outputs:
                self.teacher_14b_outputs = self.teacher_14b_outputs[:min_len]
    
    def _load_jsonl(self, filepath):
        records = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    
    def __len__(self):
        return len(self.data_samples)
    
    def __getitem__(self, idx):
        sample = self.data_samples[idx]
        text = sample[DATASET_TEXT_KEY]
        gold_summary = sample[DATASET_SUMMARY_KEY]
        
        # Tokenize article (with special tokens / BOS)
        input_enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_input_tokens,
            add_special_tokens=True,
            return_tensors=None,
        )
        
        # Tokenize gold summary (NO special tokens — raw continuation)
        target_enc = self.tokenizer(
            gold_summary,
            truncation=True,
            max_length=self.max_output_tokens,
            add_special_tokens=False,
            return_tensors=None,
        )
        
        article_ids = input_enc['input_ids']
        summary_ids = target_enc['input_ids']
        eos_id = self.tokenizer.eos_token_id
        eos_added = False
        if eos_id is not None:
            if len(summary_ids) >= self.max_output_tokens:
                summary_ids = summary_ids[:self.max_output_tokens - 1] + [eos_id]
            else:
                summary_ids = summary_ids + [eos_id]
            eos_added = True
        article_len = len(article_ids)
        summary_len = len(summary_ids)
        
        # Concatenate: [article_tokens, summary_tokens]
        # This is the correct format for decoder-only causal LM summarization
        full_input_ids = article_ids + summary_ids
        full_attention_mask = [1] * len(full_input_ids)
        
        # Labels: -100 for article positions (don't compute loss there),
        #         actual token IDs for summary positions
        full_labels = [-100] * article_len + summary_ids
        
        result = {
            'input_ids': full_input_ids,
            'attention_mask': full_attention_mask,
            'labels': full_labels,
            'summary_start': article_len,   # where summary begins
            'summary_len': summary_len,     # how many summary tokens
        }
        
        # Add teacher outputs if available
        if self.use_distillation:
            teacher_len = None
            if self.teacher_32b_outputs and idx < len(self.teacher_32b_outputs):
                result['teacher_32b_logprobs'] = self.teacher_32b_outputs[idx].get('top_k_logprobs', [])
                result['teacher_32b_token_ids'] = self.teacher_32b_outputs[idx].get('token_ids', [])
                teacher_len = len(result['teacher_32b_logprobs'])
            
            if self.teacher_14b_outputs and idx < len(self.teacher_14b_outputs):
                result['teacher_14b_logprobs'] = self.teacher_14b_outputs[idx].get('top_k_logprobs', [])
                result['teacher_14b_token_ids'] = self.teacher_14b_outputs[idx].get('token_ids', [])
                if teacher_len is None:
                    teacher_len = len(result['teacher_14b_logprobs'])
                else:
                    teacher_len = min(teacher_len, len(result['teacher_14b_logprobs']))

            if teacher_len is None:
                teacher_len = 0
            if eos_added and summary_len > 0:
                teacher_len = min(teacher_len, summary_len - 1)
            else:
                teacher_len = min(teacher_len, summary_len)

            result['teacher_mask'] = [1] * teacher_len + [0] * (summary_len - teacher_len)
        
        return result


def collate_fn(batch, tokenizer, vocab_size, use_distillation=True):
    """
    Custom collate function for concatenated [article + summary] sequences.
    
    - Left-pads input_ids and attention_mask (causal LM convention)
    - Left-pads labels with -100 (matching the left-padding)
    - Tracks summary_start per sample (adjusted for left-padding)
    - Pads teacher logprobs to max summary length
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    max_seq_len = max(len(item['input_ids']) for item in batch)
    max_summary_len = max(item['summary_len'] for item in batch)
    
    input_ids = []
    attention_masks = []
    labels = []
    summary_starts = []  # adjusted for left-padding
    teacher_masks = []
    
    for item in batch:
        seq_len = len(item['input_ids'])
        pad_len = max_seq_len - seq_len
        
        # Left-pad input_ids and attention_mask
        input_ids.append([pad_id] * pad_len + item['input_ids'])
        attention_masks.append([0] * pad_len + item['attention_mask'])
        
        # Left-pad labels with -100 (padding positions have no loss)
        labels.append([-100] * pad_len + item['labels'])
        
        # Adjust summary_start for left-padding
        summary_starts.append(item['summary_start'] + pad_len)

        if use_distillation and 'teacher_mask' in item:
            tm = item['teacher_mask']
            if len(tm) < max_summary_len:
                tm = tm + [0] * (max_summary_len - len(tm))
            else:
                tm = tm[:max_summary_len]
            teacher_masks.append(tm)
    
    result = {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long),
        'summary_starts': summary_starts,      # list of ints (per sample)
        'max_summary_len': max_summary_len,     # for teacher logprob alignment
    }

    if use_distillation and teacher_masks:
        result['teacher_mask'] = torch.tensor(teacher_masks, dtype=torch.float32)
    
    # Handle teacher logprobs — pad to max_summary_len
    if use_distillation:
        for teacher_key in ['teacher_32b_logprobs', 'teacher_14b_logprobs']:
            if teacher_key in batch[0] and batch[0][teacher_key]:
                batch_logprobs = [item.get(teacher_key, []) for item in batch]
                
                # Pad all teacher logprob sequences to max_summary_len
                padded_logprobs = []
                for seq in batch_logprobs:
                    if not seq:
                        seq = []
                    # Pad with empty entries (will become uniform in dense conversion)
                    while len(seq) < max_summary_len:
                        seq.append([])
                    seq = seq[:max_summary_len]  # Truncate if needed
                    padded_logprobs.append(seq)
                
                # Convert to dense: (batch, max_summary_len, vocab)
                dense = batch_sparse_to_dense(padded_logprobs, vocab_size)
                result[teacher_key] = dense
    
    return result


# ============================================================================
# Training Loop
# ============================================================================
def train_student(experiment_name: str, resume_from: str = None):
    """
    Train the student model for a specific experiment configuration.
    """
    assert experiment_name in EXPERIMENTS, f"Unknown experiment: {experiment_name}. Choose from: {list(EXPERIMENTS.keys())}"
    
    exp_config = EXPERIMENTS[experiment_name]
    
    print(f"\n{'='*80}")
    print(f"DUAL-TEACHER DISTILLATION — STUDENT TRAINING")
    print(f"{'='*80}")
    print(f"Experiment: {experiment_name}")
    print(f"Description: {exp_config['description']}")
    print(f"Distillation: {exp_config['use_distillation']}")
    print(f"EWAD: {exp_config['use_ewad']}")
    print(f"CPDP: {exp_config['use_cpdp']}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ===== Load tokenizer =====
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, PeftModel, TaskType
    
    print(f"\nLoading tokenizer: {STUDENT_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Use len(tokenizer) instead of tokenizer.vocab_size to include all special/added tokens
    # Qwen2.5 has token IDs beyond vocab_size (e.g., 151643 when vocab_size=151643)
    vocab_size = len(tokenizer)
    print(f"Vocab size: {vocab_size} (base: {tokenizer.vocab_size})")
    
    # ===== Load dataset =====
    print("\nLoading dataset...")
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    np.random.seed(SEED)
    indices = np.random.permutation(len(all_data))
    all_data = [all_data[i] for i in indices]
    
    total = len(all_data)
    train_end = int(TRAIN_SPLIT * total)
    val_end = train_end + int(VAL_SPLIT * total)
    
    splits = {
        'train': all_data[:train_end],
        'validation': all_data[train_end:val_end],
        'test': all_data[val_end:]
    }
    
    if MAX_SAMPLES is not None:
        for name in splits:
            splits[name] = splits[name][:MAX_SAMPLES]
    
    for name, s in splits.items():
        print(f"  {name}: {len(s)} samples")
    
    # ===== Create datasets =====
    use_distill = exp_config['use_distillation']
    
    # Determine teacher files based on experiment
    teacher_32b_train = os.path.join(TEACHER_32B_OUTPUTS, "train.jsonl") if use_distill else None
    teacher_14b_train = os.path.join(TEACHER_14B_OUTPUTS, "train.jsonl") if use_distill else None
    teacher_32b_val = os.path.join(TEACHER_32B_OUTPUTS, "validation.jsonl") if use_distill else None
    teacher_14b_val = os.path.join(TEACHER_14B_OUTPUTS, "validation.jsonl") if use_distill else None
    
    # For single-teacher experiments, nullify the unused teacher
    teacher_weights = exp_config.get("teacher_weights", None)
    if teacher_weights:
        if teacher_weights.get("32b", 0) == 0:
            teacher_32b_train = None
            teacher_32b_val = None
        if teacher_weights.get("14b", 0) == 0:
            teacher_14b_train = None
            teacher_14b_val = None
    
    print("\nCreating training dataset...")
    train_dataset = DistillationDataset(
        data_samples=splits['train'],
        teacher_32b_file=teacher_32b_train,
        teacher_14b_file=teacher_14b_train,
        tokenizer=tokenizer,
        max_input_tokens=STUDENT_MAX_INPUT_TOKENS,
        max_output_tokens=STUDENT_MAX_OUTPUT_TOKENS,
        use_distillation=use_distill,
    )
    
    print("\nCreating validation dataset...")
    val_dataset = DistillationDataset(
        data_samples=splits['validation'],
        teacher_32b_file=teacher_32b_val,
        teacher_14b_file=teacher_14b_val,
        tokenizer=tokenizer,
        max_input_tokens=STUDENT_MAX_INPUT_TOKENS,
        max_output_tokens=STUDENT_MAX_OUTPUT_TOKENS,
        use_distillation=use_distill,
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=STUDENT_BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, vocab_size, use_distill),
        num_workers=0,
        pin_memory=False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=STUDENT_BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, vocab_size, use_distill),
        num_workers=0,
        pin_memory=False,
    )
    
    # ===== Load student model with LoRA =====
    print(f"\nLoading student model: {STUDENT_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if STUDENT_BF16 else torch.float32,
        device_map="auto",
    )
    
    if STUDENT_GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
    
    # Apply or load LoRA
    if resume_from:
        if not os.path.exists(resume_from):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_from}")
        print(f"\nResuming LoRA adapter from: {resume_from}")
        model = PeftModel.from_pretrained(model, resume_from, is_trainable=True)
    else:
        print("\nApplying LoRA...")
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # ===== Loss function =====
    print("\nInitializing loss function...")
    loss_fn = DualTeacherDistillationLoss(
        vocab_size=vocab_size,
        experiment_config=exp_config,
    )
    print(f"  Mode: {'distillation' if use_distill else 'baseline CE'}")
    if use_distill and exp_config.get('use_ewad'):
        print(f"  EWAD mode: {exp_config['use_ewad']}")
    if use_distill and exp_config.get('use_cpdp'):
        print(f"  CPDP μ: {CPDP_MU}")
    
    # ===== Optimizer & Scheduler =====
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=STUDENT_LEARNING_RATE,
        weight_decay=STUDENT_WEIGHT_DECAY,
    )
    
    total_steps = len(train_loader) * STUDENT_NUM_EPOCHS // STUDENT_GRADIENT_ACCUMULATION
    warmup_steps = int(STUDENT_WARMUP_RATIO * total_steps)
    
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    print(f"\nTraining config:")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Effective batch: {STUDENT_BATCH_SIZE * STUDENT_GRADIENT_ACCUMULATION}")
    
    # ===== Output directory =====
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"{experiment_name}_{timestamp}"
    if resume_from:
        run_tag += "_resume"
    output_dir = os.path.join(STUDENT_OUTPUT_DIR, run_tag)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    
    # Save experiment config
    with open(os.path.join(output_dir, "experiment_config.json"), "w") as f:
        json.dump({
            "experiment_name": experiment_name,
            "config": exp_config,
            "student_model": STUDENT_MODEL,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "batch_size": STUDENT_BATCH_SIZE,
            "grad_accum": STUDENT_GRADIENT_ACCUMULATION,
            "lr": STUDENT_LEARNING_RATE,
            "epochs": STUDENT_NUM_EPOCHS,
            "resume_from": resume_from,
            "ewad_tau_w": EWAD_TAU_W,
            "ewad_k": EWAD_K,
            "ewad_delta": EWAD_DELTA,
            "cpdp_mu": CPDP_MU,
            "timestamp": timestamp,
        }, f, indent=2)
    
    # ===== Training Loop =====
    print(f"\n{'='*80}")
    print("STARTING TRAINING")
    print(f"{'='*80}\n")
    
    global_step = 0
    best_val_loss = float('inf')
    training_log = []
    
    for epoch in range(STUDENT_NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_diagnostics = {}
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{STUDENT_NUM_EPOCHS}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)
            summary_starts = batch['summary_starts']  # list of ints
            max_summary_len = batch['max_summary_len']
            teacher_mask = batch.get('teacher_mask', None)
            if teacher_mask is not None:
                teacher_mask = teacher_mask.to(model.device)
            
            # Forward pass on full [article + summary] sequence
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            all_logits = outputs.logits  # (B, full_seq_len, vocab)
            
            # ===== Causal shift + summary extraction =====
            # In causal LM: logits[t] predicts token[t+1]
            # For summary token at position s, the predicting logit is at position s-1
            # Teacher logprobs[j] = P(summary_token_j | prompt, summary_0..j-1)
            #   which aligns with student logits[summary_start - 1 + j]
            #
            # Extract summary-aligned logits: positions [summary_start-1, summary_start-1+summary_len)
            # We collect per-sample into a padded (B, max_summary_len, vocab) tensor
            
            B = all_logits.size(0)
            V = all_logits.size(2)
            summary_logits = torch.zeros(B, max_summary_len, V, device=model.device, dtype=all_logits.dtype)
            summary_labels = torch.full((B, max_summary_len), -100, device=model.device, dtype=torch.long)
            
            for i in range(B):
                s_start = summary_starts[i]
                # Find how many actual summary tokens this sample has
                # (count non -100 labels from s_start onward)
                sample_labels = labels[i, s_start:]  # summary portion of labels
                s_len = (sample_labels != -100).sum().item()
                if s_len == 0:
                    continue
                # Logit for summary_token_j is at position (s_start - 1 + j)
                logit_start = s_start - 1
                logit_end = logit_start + s_len
                summary_logits[i, :s_len, :] = all_logits[i, logit_start:logit_end, :]
                summary_labels[i, :s_len] = sample_labels[:s_len]
            
            # Now summary_logits[i,j] predicts summary_labels[i,j]
            # Teacher logprobs are already (B, max_summary_len, vocab) — same alignment
            student_logits = summary_logits
            aligned_labels = summary_labels
            
            # Prepare teacher logprobs (already padded to max_summary_len by collate)
            teacher_32b_lp = batch.get('teacher_32b_logprobs', None)
            teacher_14b_lp = batch.get('teacher_14b_logprobs', None)
            
            if teacher_32b_lp is not None:
                teacher_32b_lp = teacher_32b_lp.to(model.device).float()
            
            if teacher_14b_lp is not None:
                teacher_14b_lp = teacher_14b_lp.to(model.device).float()
            
            # Compute loss
            loss, diagnostics = loss_fn(
                student_logits=student_logits,
                gold_labels=aligned_labels,
                teacher_32b_logprobs=teacher_32b_lp,
                teacher_14b_logprobs=teacher_14b_lp,
                attention_mask=teacher_mask,
            )
            
            # Scale loss for gradient accumulation
            loss = loss / STUDENT_GRADIENT_ACCUMULATION
            loss.backward()
            
            epoch_loss += loss.item() * STUDENT_GRADIENT_ACCUMULATION
            num_batches += 1
            
            # Accumulate diagnostics
            for k, v in diagnostics.items():
                epoch_diagnostics[k] = epoch_diagnostics.get(k, 0) + v
            
            # Gradient step
            if (batch_idx + 1) % STUDENT_GRADIENT_ACCUMULATION == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), STUDENT_MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Logging
                if global_step % STUDENT_LOGGING_STEPS == 0:
                    avg_loss = epoch_loss / num_batches
                    lr = scheduler.get_last_lr()[0]
                    log_entry = {
                        'step': global_step,
                        'epoch': epoch + 1,
                        'loss': avg_loss,
                        'lr': lr,
                    }
                    for k, v in epoch_diagnostics.items():
                        log_entry[k] = v / num_batches
                    
                    training_log.append(log_entry)
                    progress_bar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")
                
                # Save checkpoint
                if global_step % STUDENT_SAVE_STEPS == 0:
                    ckpt_dir = os.path.join(output_dir, "checkpoints", f"step_{global_step}")
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    print(f"\n  Checkpoint saved: {ckpt_dir}")
        
        # ===== End of epoch evaluation =====
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        print(f"\nEpoch {epoch+1} — Train Loss: {avg_epoch_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"  Validation"):
                input_ids = batch['input_ids'].to(model.device)
                attention_mask = batch['attention_mask'].to(model.device)
                labels = batch['labels'].to(model.device)
                val_summary_starts = batch['summary_starts']
                val_max_summary_len = batch['max_summary_len']
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                all_logits = outputs.logits
                
                # Extract summary-aligned logits (same causal shift as training)
                B = all_logits.size(0)
                V = all_logits.size(2)
                summary_logits = torch.zeros(B, val_max_summary_len, V, device=model.device, dtype=all_logits.dtype)
                summary_labels = torch.full((B, val_max_summary_len), -100, device=model.device, dtype=torch.long)
                
                for i in range(B):
                    s_start = val_summary_starts[i]
                    sample_labels = labels[i, s_start:]
                    s_len = (sample_labels != -100).sum().item()
                    if s_len == 0:
                        continue
                    logit_start = s_start - 1
                    logit_end = logit_start + s_len
                    summary_logits[i, :s_len, :] = all_logits[i, logit_start:logit_end, :]
                    summary_labels[i, :s_len] = sample_labels[:s_len]
                
                # For validation, use CE loss only (fair comparison across experiments)
                ce_loss = F.cross_entropy(
                    summary_logits.reshape(-1, summary_logits.size(-1)),
                    summary_labels.reshape(-1),
                    ignore_index=-100
                )
                val_loss += ce_loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / max(val_batches, 1)
        print(f"  Val CE Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_dir = os.path.join(output_dir, "best_model")
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"  New best model saved! Val loss: {avg_val_loss:.4f}")
        
        training_log.append({
            'epoch_end': epoch + 1,
            'train_loss': avg_epoch_loss,
            'val_loss': avg_val_loss,
            'best_val_loss': best_val_loss,
        })
    
    # ===== Save final model =====
    final_dir = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    # Save training log
    with open(os.path.join(output_dir, "training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE — {experiment_name}")
    print(f"{'='*80}")
    print(f"Final model: {final_dir}")
    print(f"Best model: {os.path.join(output_dir, 'best_model')}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Training log: {os.path.join(output_dir, 'training_log.json')}")
    
    return output_dir


def main():
    import config as cfg
    parser = argparse.ArgumentParser(description="Train student model with distillation")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=list(EXPERIMENTS.keys()),
        help=f"Experiment config to run. Options: {list(EXPERIMENTS.keys())}"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Quick test: 1000 samples, 1 epoch (overrides config)"
    )
    args = parser.parse_args()

    if args.test_mode and not cfg.TEST_MODE:
        cfg.MAX_SAMPLES = 1000
        cfg.STUDENT_NUM_EPOCHS = 1
        cfg.STUDENT_SAVE_STEPS = 50
        cfg.STUDENT_LOGGING_STEPS = 10
        cfg.STUDENT_GRADIENT_ACCUMULATION = 2
        cfg.TEACHER_32B_OUTPUTS = os.path.join(cfg.BASE_DIR, "teacher_outputs_test", "teacher_32b")
        cfg.TEACHER_14B_OUTPUTS = os.path.join(cfg.BASE_DIR, "teacher_outputs_test", "teacher_14b")
        cfg.STUDENT_OUTPUT_DIR = os.path.join(cfg.BASE_DIR, "student_outputs_test")
        # Update module-level names used by train_student()
        for k in ['MAX_SAMPLES', 'STUDENT_NUM_EPOCHS', 'STUDENT_SAVE_STEPS',
                  'STUDENT_LOGGING_STEPS', 'STUDENT_GRADIENT_ACCUMULATION',
                  'TEACHER_32B_OUTPUTS', 'TEACHER_14B_OUTPUTS', 'STUDENT_OUTPUT_DIR']:
            globals()[k] = getattr(cfg, k)
        print("\n*** --test-mode: 1000 samples, 1 epoch, test output dirs ***\n")

    output_dir = train_student(args.experiment, resume_from=args.resume)
    print(f"\nDone. Output: {output_dir}")


if __name__ == "__main__":
    main()
