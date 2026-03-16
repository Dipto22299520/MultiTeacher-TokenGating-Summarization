# BanglaT5 MapReduce Summarization System — Complete Technical Documentation

A sophisticated two-stage **MapReduce summarization pipeline** for long Bangla news articles using fine-tuned **BanglaT5 (296M parameter)** transformer models. This project addresses the fundamental challenge of summarizing documents that exceed the model's 1024-token context window by splitting them into manageable chunks, summarizing each independently, and merging the results.

**Hardware Used**: NVIDIA RTX 5080 (16 GB VRAM)  
**Base Model**: `csebuetnlp/banglaT5` (T5 architecture, 296M parameters)  
**Dataset**: XL-Sum Bangla (BBC Bangla news articles)  
**Status**: Research Project — Completed Training & Evaluation  
**Last Updated**: March 2026

---

## Table of Contents

1. [Problem Statement & Motivation](#1-problem-statement--motivation)
2. [Architecture Overview](#2-architecture-overview)
3. [The Four Solutions for Long Documents](#3-the-four-solutions-for-long-documents)
4. [Dataset & Data Processing](#4-dataset--data-processing)
5. [Training Pipeline — Detailed Walkthrough](#5-training-pipeline--detailed-walkthrough)
   - [5.1 Stage 0: Production/Teacher Model (Short Articles)](#51-stage-0-productionteacher-model-short-articles)
   - [5.2 Stage 1: Chunked Model Training](#52-stage-1-chunked-model-training)
   - [5.3 Stage 2: Full Document Model (MAP Model)](#53-stage-2-full-document-model-map-model)
   - [5.4 Stage 3: Reduce Task Model](#54-stage-3-reduce-task-model)
6. [Inference Pipeline — How Summarization Works](#6-inference-pipeline--how-summarization-works)
7. [Evaluation System](#7-evaluation-system)
8. [Results & Performance](#8-results--performance)
9. [Complete File Reference](#9-complete-file-reference)
10. [Key Design Decisions & Lessons Learned](#10-key-design-decisions--lessons-learned)
11. [Utility Scripts Reference](#11-utility-scripts-reference)
12. [Configuration Reference](#12-configuration-reference)
13. [Installation & Quick Start](#13-installation--quick-start)
14. [Known Issues & Solutions](#14-known-issues--solutions)
15. [Research Contributions](#15-research-contributions)
16. [References](#16-references)

---

## 1. Problem Statement & Motivation

### The Core Challenge

Standard transformer models like T5 have a fixed context window — for BanglaT5, this is **1024 BPE tokens**. Many Bangla news articles from BBC Bangla exceed this limit significantly (some reaching thousands of tokens). Simply truncating the input loses critical information from the latter parts of articles.

### The Quality Problem (ROUGE Trap)

The initial approach of training on chunked text produced models with **good ROUGE scores (~0.465)** but **mediocre human-perceived quality**. The root cause was identified as:

- **Missing global salience learning**: The model learned to "compress whatever text you see" instead of understanding what's important across the full document
- **Missing importance ranking**: No ability to decide which events dominate the narrative
- **Untrained reduce phase**: The hierarchical pipeline's merge step was operating zero-shot — the model was never trained on the `chunk_summaries → final_summary` mapping
- **Dataset distribution shift**: Training on ~1,075 short article chunks caused overfitting to patterns like lead bias, entity copying, and sentence compression

### The Solution: MapReduce Architecture

The project implements a **two-model MapReduce** architecture:

1. **MAP Model** (Full Document Model): Trained on full documents to learn **global salience** — what's important, which events matter, what to ignore
2. **REDUCE Model**: Trained on teacher-generated chunk summaries to learn **merging, deduplication, and coherence synthesis**

This is analogous to how humans read: we don't process a document in perfect chunks — we reread, keep mental summaries, and tolerate redundancy.

---

## 2. Architecture Overview

### Two-Stage Pipeline Flow

```
Long Article (>1024 tokens)
    │
    ▼
┌─────────────────────────────────────────┐
│  SENTENCE SPLITTER                       │
│  (bangla_sentence_splitter.py)           │
│  - Rule-based Bangla sentence segmenter  │
│  - Handles: ।  ॥  ?  !  and English .   │
│  - Respects 22 Bangla abbreviations      │
│  - Handles quoted text boundaries        │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  SENTENCE-ALIGNED CHUNKER               │
│  (chunk_processor.py)                    │
│  - Max ~900 tokens per chunk             │
│  - 3 sentences overlap between chunks    │
│  - Never breaks mid-sentence             │
│  - Min 50 tokens per chunk               │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  MAP PHASE                               │
│  Model: checkpoint-7000 (Full Doc)       │
│  Prefix: "summarize bangla news: "       │
│  Each chunk → individual summary         │
│  - max_length: 256, min_length: 40       │
│  - num_beams: 5, length_penalty: 1.2     │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  SENTENCE DEDUPLICATION                  │
│  Jaccard similarity > 0.75 → remove      │
│  Removes redundant sentences across      │
│  chunk summaries                         │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  REDUCE PHASE                            │
│  Model: checkpoint-6000 (Reduce Task)    │
│  Prefix: "summarize multiple summaries: "│
│  Concatenated summaries → final summary  │
│  - max_length: 256, min_length: 80       │
│  - num_beams: 5, length_penalty: 2.0     │
│  - Recursive reduce if > 900 tokens      │
└─────────────────┬───────────────────────┘
                  │
                  ▼
           Final Summary
```

### Three-Mode Routing Strategy

The system automatically selects the best approach based on article length:

| Mode | Condition | Description |
|------|-----------|-------------|
| **SINGLE_PASS** | < 900 tokens | Direct summarization, no chunking needed |
| **SINGLE_CHUNK** | 900–1024 tokens | One chunk, MAP only, no REDUCE needed |
| **MAP_REDUCE** | > 1024 tokens | Full MapReduce pipeline |

On the test set (1,985 articles): Single Pass: 341 (17.2%), Single Chunk: 165 (8.3%), MapReduce: 1,479 (74.5%)

### Model Components

| Component | Checkpoint Path | Base Model | Purpose | Training Data |
|-----------|----------------|------------|---------|---------------|
| **MAP Model** | `banglaT5_full_doc_20260215_123349/checkpoint-7000` | `csebuetnlp/banglaT5` | Extract salient facts from each chunk | Full documents (79,502 samples, first 1024 tokens → gold summary) |
| **REDUCE Model** | `banglaT5_reduce_task_20260217_111025/checkpoint-6000` | MAP checkpoint-7000 (transfer learning) | Merge and deduplicate chunk summaries | Teacher-generated chunk summaries → gold summary |

---

## 3. The Four Solutions for Long Documents

The project implements four complementary solutions for handling documents that exceed the 1024-token limit:

### Solution 1: Sentence-Aligned Chunking (ENABLED — Core)

**File**: `chunk_processor.py`

Never chunks by raw token count. Instead:
1. Split text into sentences using rule-based Bangla sentence segmenter
2. Accumulate sentences until the next sentence would exceed 900 BPE tokens
3. Stop before overflow, start a new chunk at the sentence boundary

**Parameters**:
- `max_tokens = 900` — leaves room for prefix tokens + special tokens within 1024
- `min_chunk_tokens = 50` — prevents tiny trailing chunks
- Giant single sentences (> 900 tokens) become standalone chunks

**Why 900 and not 1024?**: The input prefix `"summarize bangla news: "` consumes ~8 tokens, plus special tokens (`<s>`, `</s>`), plus safety margin.

### Solution 2: Sliding Overlap (ENABLED — Core)

**File**: `chunk_processor.py`

Adjacent chunks share their last 3 sentences (~10-15% token overlap):

```
Chunk 1: S1 S2 S3 S4 S5
Chunk 2:          S3 S4 S5 S6 S7 S8
Chunk 3:                   S6 S7 S8 S9 S10
```

**Parameter**: `overlap_sentences = 3`

**Why overlap is beneficial**: 
- Discourse often spans multiple sentences
- The model sees how ideas end and how new ones begin
- Stabilizes memory across chunk boundaries
- The duplication is intentional and helpful — it prevents information loss at boundaries

### Solution 3: Memory-Aware Chunk Headers (AVAILABLE — Used in inference_pipeline.py)

**File**: `memory_header.py`

Before each chunk (except the first), inject a short summary of previous chunks:

```
[পূর্ববর্তী সারাংশ: Previous context summary here...]
[CHUNK CONTENT]
```

**Parameters**:
- `MAX_MEMORY_TOKENS = 100` — budget for memory header
- `MIN_CONTENT_TOKENS = 600` — minimum remaining for actual content
- Memory prefix: `"পূর্ববর্তী সারাংশ:"` ("Previous summary:" in Bangla)

**Two modes**:
- **Training**: Extractive memory — first sentence of each previous chunk
- **Inference**: Model-generated memory — uses actual summaries from prior chunks

### Solution 4: Chunk-Aware Attention Bias (AVAILABLE — NOT ENABLED in final models)

**File**: `attention_bias.py`

Advanced technique: instead of hard chunk boundaries, apply a learnable attention bias in the T5 encoder. Tokens within the same chunk attend more strongly; cross-boundary attention is dampened.

**Parameters**:
- `alpha = 0.5` — dampening strength
- `decay_type = 'linear'` or `'exponential'`
- `learnable = True` — alpha is a learnable parameter

**Why it was disabled**: Small models (296M parameters) get confused by multiple architectural signals. The simpler solutions (1-3) were sufficient.

---

## 4. Dataset & Data Processing

### Source Dataset

**XL-Sum Bangla** — Cross-lingual summarization dataset from BBC Bangla news articles.

| File | Description | Samples |
|------|-------------|---------|
| `xlsum_all_train.csv` | Raw XL-Sum training data (CSV) | ~79,502 |
| `bangla_train_combined.json` | Combined dataset (all articles) | ~79,502 |
| `bangla_train_lte_1000.json` | Articles ≤1000 BPE tokens (short) | ~7,027 |
| `bangla_train_gt_1000.json` | Articles >1000 BPE tokens (long) | Remaining |
| `bansum_over_1000_tokens.json` | Long articles collection | Variable |

### Data Splitting Strategy

**File**: `split_dataset.py`

The combined dataset is split with a fixed random seed (42) to ensure reproducibility:

| Split | Ratio | Output File | Samples |
|-------|-------|-------------|---------|
| Training | 80% | `data_splits/train.json` | ~63,602 |
| Validation | 10% | `data_splits/val.json` | ~7,950 |
| Test | 10% | `data_splits/test.json` | ~7,950 (1,988 used in eval) |

### Critical Data Processing: Bangla Character Fix

**Function**: `fix_bangla_for_tokenizer(text)` — used across multiple scripts

BanglaT5's SentencePiece vocabulary does not contain precomposed Bangla characters with nukta (়). These characters map to `<unk>` tokens, which degrades model performance.

**The fix decomposes**:
- `য়` (U+09DF) → `য` + `়` (U+09AF + U+09BC)
- `ড়` (U+09DC) → `ড` + `়` (U+09A1 + U+09BC)
- `ঢ়` (U+09DD) → `ঢ` + `়` (U+09A2 + U+09BC)

**Also normalizes**: Smart quotes → ASCII quotes, em/en dashes → hyphens, ellipsis → three dots, removes ZWNJ (U+200C) and ZWJ (U+200D)

**Verification**: `verify_data.py` checks for precomposed characters, empty fields, and UNK token rates across all splits.

### Reduce Task Data Generation

**File**: `generate_reduce_data.py`

Creates training data for the REDUCE model using the trained MAP model (checkpoint-7000) as a teacher:

1. For each article: chunk it → generate per-chunk summaries using the teacher model → concatenate with `[CHUNK]` markers
2. Input: concatenated teacher-generated chunk summaries
3. Target: original gold summary

**Data augmentation** (training split only):
| Augmentation | Description | Purpose |
|-------------|-------------|---------|
| `clean` | Original chunk order | Baseline |
| `shuffle` | Random chunk order (if ≥2 chunks) | Teaches order-invariance |
| `drop` | One random chunk removed (if ≥3 chunks) | Teaches missing-info tolerance |
| `duplicate` | One sentence duplicated within a chunk | Teaches deduplication |

**Quality filters**: text ≥100 chars, summary ≥20 chars, text ≥30 words, summary ≥8 words, summary ≤150 words

**Output files**: `reduce_data/reduce_train.json`, `reduce_data/reduce_val.json`, `reduce_data/reduce_test.json`

**Teacher generation parameters**: max_length=256, min_length=30, num_beams=4, batch_size=16

---

## 5. Training Pipeline — Detailed Walkthrough

### Evolution of Training Approaches

The project evolved through four training stages, each addressing limitations of the previous:

```
Stage 0: Production Model (short articles only, ≤1000 tokens)
    ↓ Problem: Only handles short articles
Stage 1: Chunked Model (long articles with Solutions 1-3)
    ↓ Problem: Learns local compression, not global salience
Stage 2: Full Document Model (all articles, truncated to 1024 tokens)
    ↓ Problem: Reduce phase is zero-shot/untrained
Stage 3: Reduce Task Model (merges chunk summaries)
    ↓ Result: Full trained MapReduce pipeline
```

---

### 5.1 Stage 0: Production/Teacher Model (Short Articles)

**Script**: `train_bangla_teacher.py`  
**Output Directory**: `banglaT5_production_20260210_131619/`  
**Purpose**: Baseline fine-tune on short articles (≤1000 tokens). The "teacher" model.

#### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base Model | `csebuetnlp/banglaT5` |
| Data File | `bangla_train_lte_1000.json` |
| Input Prefix | `"summarize: "` |
| Max Input Length | 1024 tokens |
| Max Target Length | 192 tokens |
| Batch Size | 4 (effective: 32 with gradient accumulation 8) |
| Epochs | 15 |
| Learning Rate | 5e-5 |
| Warmup Ratio | 0.1 |
| Weight Decay | 0.01 |
| Label Smoothing | 0.1 |
| Num Beams | 4 |
| Length Penalty | 1.0 |
| No Repeat N-gram Size | 2 |
| Repetition Penalty | 1.2 |
| Early Stopping Patience | 5 |
| Best Model Metric | `eval_loss` (lower is better) |
| Gradient Checkpointing | OFF |
| bf16 | Auto-detected |
| Evaluation During Training | Loss only (no generation) |

#### Data Split
- Train: 5,972 samples (85%)
- Validation: 703 samples (10%)
- Test: 352 samples (5%)
- Split method: `sklearn.train_test_split` with seed 42

#### Key Design Decisions
- **English prefix `"summarize: "`**: Found to converge faster than Bangla prefix
- **No generation during training**: Only loss computed for speed. ROUGE evaluated post-training only
- **Label smoothing 0.1**: Prevents overconfident decoding
- **Character-level ROUGE**: Standard for morphologically-rich languages like Bangla

#### Results

| Metric | Validation | Test |
|--------|-----------|------|
| Loss | 3.9759 | 3.9924 |
| ROUGE-1 | 0.6892 | 0.7003 |
| ROUGE-2 | 0.4654 | 0.4781 |
| ROUGE-L | **0.4588** | **0.4648** |

**Limitation**: Only trained on short articles. Cannot handle long documents. ROUGE scores high but human-perceived quality mediocre due to local compression learning.

---

### 5.2 Stage 1: Chunked Model Training

**Script**: `train_bangla_chunked.py`  
**Output Directory**: `banglaT5_chunked_20260213_193538/`  
**Purpose**: Handle long articles (>1000 tokens) using Solutions 1-3.

#### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base Model | `csebuetnlp/banglaT5` |
| Data Source | Pre-split JSONL (`data_splits/train.jsonl`, etc.) |
| Input Prefix | `"summarize bangla news: "` |
| Max Input Length | 1024 tokens |
| Max Target Length | 256 tokens |
| Min Target Length | 64 tokens |
| Batch Size | 4 (effective: 32) |
| Epochs | 25 |
| Learning Rate | 3e-5 |
| Warmup Ratio | 0.1 |
| Weight Decay | 0.01 |
| Label Smoothing | 0.05 |
| Num Beams | 6 |
| Length Penalty | 1.2 |
| No Repeat N-gram Size | 3 |
| Repetition Penalty | 1.15 |
| Attention Bias (Solution 4) | **DISABLED** (`USE_ATTENTION_BIAS = False`) |
| Early Stopping Patience | 5 |
| Best Model Metric | `eval_rougeL` (higher is better) |
| Evaluation Strategy | Every 500 steps WITH generation |
| Gradient Checkpointing | OFF |

#### Special Features
- **Chunk metadata injection**: Input prepended with `[chunk X/Y | with prior context]`
- **Character-level ROUGE** for Bangla with custom `CharTokenizer`
- **Repetition rate** and **length ratio** tracked as custom metrics
- **Optional `ChunkAwareT5`** wrapper (Solution 4, disabled)

#### Data Cleaning
Filters applied per split:
- Text < 80 chars → removed
- Summary < 16 chars → removed
- Text < 20 words → removed
- Summary < 5 words → removed
- Summary > 140 words → removed
- Deduplication by (text, summary) pair

#### Results

| Metric | Validation | Test |
|--------|-----------|------|
| Loss | 4.2501 | 4.3204 |
| ROUGE-1 | 0.6073 | 0.5691 |
| ROUGE-2 | 0.3901 | 0.3745 |
| ROUGE-L | **0.3636** | **0.3461** |
| Gen Length | 79.71 | 78.40 |
| Repetition Rate | 0.1805 | 0.1746 |

**Training Data**: 1,888 train / 221 val / 105 test samples

**Limitation**: Lower ROUGE than Stage 0 because training on chunks teaches local compression, not global understanding of what's important. This led to the insight about the "ROUGE trap."

---

### 5.3 Stage 2: Full Document Model (MAP Model)

**Script**: `train_full_document.py`  
**Output Directory**: `banglaT5_full_doc_20260215_123349/`  
**Best Checkpoint**: `checkpoint-7000`  
**Purpose**: **THE MOST IMPORTANT STEP** — teaches the model global salience learning by training on full documents.

#### The Key Insight

The first 1024 tokens of most Bangla news articles contain:
- Main entities
- Main events
- Core narrative

By training `full_document[:1024] → gold_summary`, the model learns:
- **Salience ranking** — what's important
- **Discourse compression** — how to condense narrative
- **Abstraction patterns** — how to rephrase
- **Narrative structure** — how articles are organized

This is the **missing capability** from previous approaches.

#### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base Model | `csebuetnlp/banglaT5` (CLI-overridable) | |
| Input Prefix | `"summarize bangla news: "` | |
| Max Input Length | 1024 tokens | Full context window |
| Max Target Length | 256 tokens | Longer summaries than Stage 0 |
| Min Target Length | 64 tokens | |
| Batch Size | 4 (effective: 32) | |
| Epochs | 15 | |
| Learning Rate | 3e-5 | |
| Warmup Ratio | 0.1 | |
| Weight Decay | 0.01 | |
| Label Smoothing | **0.0** | Hard targets — forces exact copy of gold summary |
| Repetition Penalty | **1.0** | No penalty — allows extractive/copy patterns |
| Num Beams | 5 | |
| Length Penalty | 1.0 | |
| No Repeat N-gram Size | 3 | |
| Gradient Checkpointing | **ON** | Memory efficiency for large dataset |
| bf16 | **True** (hardcoded) | |
| Evaluation During Training | **NONE** | Saves all checkpoints, evaluates later |
| Save All Checkpoints | Yes (saves every 500 steps) | |

#### Critical Code Feature: `fix_bangla_for_tokenizer()`

This script introduced the Bangla character decomposition fix that prevents `<unk>` tokens from precomposed characters (য়, ড়, ঢ়). Applied to both input and target during preprocessing.

#### Tokenizer Strategy

Uses slow tokenizer (`T5Tokenizer` with `use_fast=False`) to avoid tiktoken/fast conversion issues with Bangla text. Falls back to `AutoTokenizer` if slow tokenizer fails.

#### CLI Arguments

```bash
python train_full_document.py \
    --train_file data_splits/train.json \
    --val_file data_splits/val.json \
    --test_file data_splits/test.json \
    --model csebuetnlp/banglaT5 \
    --batch_size 4 \
    --epochs 15 \
    --lr 3e-5 \
    --resume_from_checkpoint <path>  # Optional: resume training
    --output_dir <path>               # Optional: custom output dir
```

#### Checkpoint Evaluation Results

All 14 checkpoints were evaluated on 1,000 test samples:

| Checkpoint | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic Sim. | BERTScore F1 | BLEU |
|-----------|---------|---------|---------|---------------|--------------|------|
| cp-500 | 0.1564 | 0.0360 | 0.0913 | 0.6777 | 0.6812 | 6.88 |
| cp-1000 | 0.1971 | 0.0522 | 0.1108 | 0.6959 | 0.6947 | 12.36 |
| cp-1500 | 0.1990 | 0.0550 | 0.1140 | 0.7057 | 0.6989 | 7.88 |
| cp-2000 | 0.2213 | 0.0962 | 0.1563 | 0.7417 | 0.7228 | 7.69 |
| cp-2500 | 0.2253 | 0.1005 | 0.1620 | 0.7452 | 0.7245 | 11.59 |
| cp-3000 | 0.2352 | 0.1104 | 0.1759 | 0.7548 | 0.7287 | 8.96 |
| cp-3500 | 0.2387 | 0.1128 | 0.1787 | 0.7501 | 0.7284 | 8.45 |
| cp-4000 | 0.2397 | 0.1141 | 0.1800 | 0.7497 | 0.7294 | 5.39 |
| cp-4500 | 0.2384 | 0.1167 | 0.1820 | 0.7501 | 0.7290 | 5.55 |
| cp-5000 | 0.2423 | 0.1168 | 0.1834 | 0.7489 | 0.7302 | 6.90 |
| cp-5500 | 0.2424 | 0.1176 | 0.1851 | 0.7509 | 0.7296 | 5.85 |
| cp-6000 | 0.2437 | 0.1186 | 0.1852 | 0.7519 | 0.7305 | 15.04 |
| cp-6500 | 0.2445 | 0.1197 | 0.1871 | 0.7526 | 0.7306 | 6.63 |
| **cp-7000** | **0.2480** | **0.1216** | **0.1885** | **0.7542** | **0.7314** | **15.98** |

**Best checkpoint: checkpoint-7000** — best across all metrics

**Note on ROUGE scores**: These appear lower than Stage 0/1 because they use **word-level ROUGE** (not char-level). Char-level ROUGE inflates scores for morphologically-rich languages. The semantic metrics (BERTScore 0.731, Semantic Similarity 0.754) show strong performance. The model produces more abstractive, coherent summaries.

---

### 5.4 Stage 3: Reduce Task Model

**Script**: `train_reduce_task.py`  
**Output Directory**: `banglaT5_reduce_task_20260217_111025/`  
**Best Checkpoint**: `checkpoint-6000`  
**Purpose**: Train the model to merge multiple chunk summaries into a coherent final summary.

#### The Problem This Solves

At inference time, the reduce phase receives:
```
model_generated_chunk_summaries → final_summary
```

But without explicit training, the model was operating **zero-shot** on this mapping. Teacher-generated chunk summaries are noisy, redundant, partially wrong, and stylistically inconsistent — completely different from gold summary fragments. This is the **train/inference distribution mismatch** problem.

#### Key Design: Transfer Learning from MAP Model

The REDUCE model starts from the **MAP model's checkpoint-7000** (not from the pretrained HuggingFace model). This gives it:
- Knowledge of Bangla summarization patterns
- Understanding of salience
- Familiarity with the domain

Then it's fine-tuned specifically on the reduce task.

#### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base Model | `./banglaT5_full_doc_20260215_123349/checkpoint-7000` |
| Input Prefix | `"summarize multiple summaries: "` |
| Max Input Length | 1024 tokens |
| Max Target Length | 256 tokens |
| Batch Size | 4 (effective: 32) |
| Epochs | 15 |
| Learning Rate | 3e-5 |
| Warmup Ratio | 0.1 |
| Weight Decay | 0.01 |
| Label Smoothing | 0.0 |
| Gradient Checkpointing | ON |
| bf16 | True |
| Evaluation During Training | NONE |
| Save All Checkpoints | Yes (every 500 steps) |

#### CLI Arguments

```bash
python train_reduce_task.py \
    --data_dir reduce_data \
    --model ./banglaT5_full_doc_20260215_123349/checkpoint-7000 \
    --batch_size 4 \
    --epochs 15 \
    --lr 3e-5 \
    --resume_from <checkpoint_path>  # Optional
```

#### Why Different Input Prefix Matters

The reduce model uses `"summarize multiple summaries: "` instead of `"summarize bangla news: "`. This signals to the model that the input is already summarized content that needs merging, not raw article text. The different prefix creates a separate "task space" within the same model architecture.

---

## 6. Inference Pipeline — How Summarization Works

### Primary Pipeline: `run_pipeline.py`

This is the **main production pipeline** used for inference and evaluation.

#### Class: `ChunkedSummarizer`

```python
from run_pipeline import ChunkedSummarizer

summarizer = ChunkedSummarizer(
    map_model_path="./banglaT5_full_doc_20260215_123349/checkpoint-7000",
    reduce_model_path="./banglaT5_reduce_task_20260217_111025/checkpoint-6000"
)

result = summarizer.summarize("আপনার বাংলা নিবন্ধ এখানে...")
print(result['summary'])
print(f"Method: {result['method']}")      # single_pass / single_chunk / map_reduce
print(f"Chunks: {result['num_chunks']}")
print(f"Tokens: {result['input_tokens']}")
```

#### Generation Parameters (by phase)

| Parameter | MAP Phase | REDUCE Phase | Single-Pass |
|-----------|-----------|-------------|-------------|
| max_length | 256 | 256 | 256 |
| min_length | 40 | 80 | 60 |
| num_beams | 5 | 5 | 5 |
| length_penalty | 1.2 | **2.0** | 1.5 |
| no_repeat_ngram_size | 3 | 3 | 3 |
| repetition_penalty | 1.0 | 1.0 | 1.0 |
| early_stopping | True | True | True |

**Note**: REDUCE has higher `length_penalty` (2.0) to encourage longer, more comprehensive merged summaries.

#### Sentence Deduplication

After MAP phase, before REDUCE, redundant sentences are removed using Jaccard word-overlap similarity:
- Threshold: 0.75 (if 75%+ of words overlap, sentences are considered duplicates)
- Only the first occurrence is kept

#### Recursive Reduce

If the concatenated chunk summaries exceed 900 tokens, the REDUCE phase is applied recursively: first reduce subgroups, then reduce the group summaries.

### Alternative Pipeline: `inference_pipeline.py`

An earlier pipeline with additional features:

- **Memory headers** (Solution 3) enabled by default
- **Chunk-aware attention bias** detection (Solution 4)
- Different prompting: MAP uses `"summarize bangla news: {text}"`, REDUCE uses `"summarize bangla news: write main themes and key points from: {text}"`
- Diversity penalty (0.3) in beam search
- Jaccard dedup threshold: 0.8 (stricter)
- Auto-discovers model directories

---

## 7. Evaluation System

### Metrics Used

| Metric | Type | Implementation | Purpose |
|--------|------|----------------|---------|
| **ROUGE-1** | Surface overlap | `rouge_score` library | Unigram overlap with reference |
| **ROUGE-2** | Surface overlap | `rouge_score` library | Bigram overlap with reference |
| **ROUGE-L** | Surface overlap | `rouge_score` library | Longest common subsequence |
| **BERTScore F1** | Semantic | `bert_score` library, lang=`bn` | Contextual embedding similarity |
| **Semantic Similarity** | Semantic | `sentence-transformers`, cosine similarity | Sentence-level semantic match |
| **BLEU** | Surface overlap | `sacrebleu` / `nltk` | N-gram precision |
| **BARTScore** | Model-based | `facebook/mbart-large-cc25` | Learned evaluation metric |

### ROUGE Tokenization: Character vs Word Level

A critical implementation detail: **different scripts use different ROUGE tokenization**:

| Script | ROUGE Tokenization | Reason |
|--------|-------------------|--------|
| `train_bangla_teacher.py` | **Character-level** | Standard for Bangla |
| `train_bangla_chunked.py` | **Character-level** | Consistent with teacher |
| `train_full_document.py` | **Word-level** | Standard comparison |
| `train_reduce_task.py` | **Word-level** | Consistent with full doc |
| `eval_pipeline.py` | **Word-level** | Standard comparison |
| `eval_checkpoint.py` | **Character-level** | Character-level analysis |

**Character-level ROUGE** scores are higher (~0.46) than **word-level** (~0.19) for the same output because Bangla words share many characters (morphological similarity inflates scores).

### Semantic Embedding Models Used

| Model | Used In |
|-------|---------|
| `paraphrase-multilingual-mpnet-base-v2` | Pipeline evaluation, checkpoint evaluation, full article evaluation |
| `paraphrase-multilingual-MiniLM-L12-v2` | Remaining 388 samples evaluation |

### Evaluation Scripts

| Script | Purpose | Scope |
|--------|---------|-------|
| `eval_pipeline.py` | Sequential pipeline evaluation | Full test set, one-by-one |
| `eval_pipeline_batched.py` | **Batched GPU** pipeline evaluation (faster) | Full test set, batched |
| `eval_checkpoint.py` | Evaluate any single checkpoint | Single checkpoint |
| `eval_reduce_checkpoints.py` | Compare all reduce checkpoints | All reduce checkpoints |
| `eval_remaining_388.py` | Evaluate remaining test samples | Indices 1600–1987 |
| `evaluate_full_articles.py` | Comprehensive 5-metric evaluation | Any model |
| `check_point_checker.py` | Evaluate all MAP checkpoints | All full-doc checkpoints |
| `quick_eval.py` | Fast visual quality check (5 samples) | Quick validation |
| `compare_models.py` | Side-by-side old vs new model | Two models |
| `sample_predictions.py` | Visual output inspection (10 samples) | Quick diagnosis |

---

## 8. Results & Performance

### Full MapReduce Pipeline Results (Best Configuration)

Evaluated on **1,985 test articles** using batched evaluation:

| Metric | Score |
|--------|-------|
| **ROUGE-1** | 0.2193 |
| **ROUGE-2** | 0.0984 |
| **ROUGE-L** | 0.1628 |
| **BERTScore F1** | **0.7250** |
| **Semantic Similarity** | **0.7266** |
| Generation Time | 4,377.8 seconds (2.21 s/article) |
| MAP Time | 3,155.7 seconds |
| REDUCE Time | 1,220.0 seconds |
| Avg Chunks per Article | 5.87 |
| Max Chunks | 2,576 (outlier) |
| Avg Predicted Words | 72.9 |
| Avg Reference Words | 114.3 |

### Interpreting the Results

**ROUGE scores appear low** compared to the teacher model — this is expected because:

1. **Word-level vs character-level**: Pipeline uses word-level ROUGE (lower by ~50%)
2. **Abstractive vs extractive**: The full-doc model generates more abstractive summaries that use different words but capture the same meaning
3. **Long vs short articles**: Pipeline handles 1000+ token articles which are inherently harder to summarize

**BERTScore (0.725) and Semantic Similarity (0.727)** are strong indicators of quality — the model captures the meaning even when using different words.

### Model Evolution Summary

| Model | ROUGE-L | Type | Notes |
|-------|---------|------|-------|
| Production (teacher) | 0.465 (char) | Short articles only | Good ROUGE, mediocre quality |
| Chunked | 0.346 (char) | Long articles (chunked) | Lower ROUGE, local compression |
| Full Document cp-7000 | 0.189 (word) | All articles | Global salience, abstractive |
| Pipeline (MAP+REDUCE) | 0.163 (word) | Full system | End-to-end on long articles |

---

## 9. Complete File Reference

### Core Pipeline Files

| File | Purpose |
|------|---------|
| `run_pipeline.py` | **Main inference pipeline** — ChunkedSummarizer class, MAP+REDUCE |
| `inference_pipeline.py` | Alternative pipeline with memory headers and attention bias support |
| `bangla_sentence_splitter.py` | Rule-based Bangla sentence segmenter with BPE token counter |
| `chunk_processor.py` | Sentence-aligned chunking with sliding overlap |
| `memory_header.py` | Memory-aware chunk headers (Solution 3) |
| `attention_bias.py` | Chunk-aware attention bias (Solution 4) |

### Training Scripts

| File | Purpose |
|------|---------|
| `train_bangla_teacher.py` | Stage 0: Baseline teacher model on short articles |
| `train_bangla_chunked.py` | Stage 1: Chunked model with Solutions 1-3 |
| `train_full_document.py` | Stage 2: Full document model (MAP model) |
| `train_reduce_task.py` | Stage 3: Reduce task model |

### Data Processing Scripts

| File | Purpose |
|------|---------|
| `split_dataset.py` | Split combined dataset into train/val/test with Bangla char fix |
| `generate_reduce_data.py` | Generate reduce task training data using teacher model |
| `prepare_gt1000_training_data.py` | Process long articles through chunking pipeline for training |
| `verify_data.py` | Data integrity verification (precomposed chars, empty fields, UNK rates) |
| `extract_remaining_samples.py` | Extract test samples at indices 1600–1987 for separate evaluation |
| `_check_reduce.py` | Quick diagnostic: inspect reduce data structure and augmentation breakdown |

### Evaluation Scripts

| File | Purpose |
|------|---------|
| `eval_pipeline.py` | Sequential full-test-set pipeline evaluation |
| `eval_pipeline_batched.py` | Batched GPU pipeline evaluation (faster) |
| `eval_checkpoint.py` | Evaluate single checkpoint with Seq2SeqTrainer |
| `eval_reduce_checkpoints.py` | Compare all reduce task checkpoints |
| `eval_remaining_388.py` | Evaluate remaining 388 test samples |
| `evaluate_full_articles.py` | Comprehensive 5-metric evaluation |
| `check_point_checker.py` | Evaluate all MAP model checkpoints, recommend cleanup |
| `quick_eval.py` | Fast 5-sample visual quality check |
| `compare_models.py` | Side-by-side model comparison |
| `sample_predictions.py` | Generate 10 sample predictions for inspection |
| `# Sample 5 predictions to see quality.py` | Quick 3-sample visual check (earlier script) |

### Documentation

| File | Purpose |
|------|---------|
| `README.md` | This comprehensive documentation |
| `README_QUICKSTART.md` | Quick-start guide for training |
| `IMPROVEMENT_GUIDE.md` | Detailed theory on improvement approach |
| `chatgpt_suggestion_to_improve_quality.txt` | Expert suggestions on architecture and training |
| `suggestion.txt` | Chunking solutions and boundary diagnostics |
| `suggeston.txt` | Sample predictions from earlier pipeline (visual quality check) |

### Data Files

| File | Format | Purpose |
|------|--------|---------|
| `xlsum_all_train.csv` | CSV | Raw XL-Sum Bangla training data |
| `bangla_train_combined.json` | JSON | All articles combined (~79,502) |
| `bangla_train_lte_1000.json` | JSON | Short articles (≤1000 tokens) |
| `bangla_train_gt_1000.json` | JSON | Long articles (>1000 tokens) |
| `bansum_over_1000_tokens.json` | JSON | Long articles collection |
| `data_splits/train.json` | JSON | Training split (80%) |
| `data_splits/val.json` | JSON | Validation split (10%) |
| `data_splits/test.json` | JSON | Test split (10%) |
| `reduce_data/reduce_train.json` | JSON | Reduce task training data |
| `reduce_data/reduce_val.json` | JSON | Reduce task validation data |
| `reduce_data/reduce_test.json` | JSON | Reduce task test data |
| `test_remaining_388.json` | JSON | Remaining 388 test samples |
| `sample.txt` | Text | Sample reduce input/output |

### Model Directories

| Directory | Stage | Best Checkpoint | Key Results |
|-----------|-------|-----------------|-------------|
| `banglaT5_production_20260210_131619/` | Stage 0: Teacher | `checkpoint-2500` / `final_model` | ROUGE-L: 0.465 (char) |
| `banglaT5_chunked_20260213_193538/` | Stage 1: Chunked | `checkpoint-1475` / `final_model` | ROUGE-L: 0.346 (char) |
| `banglaT5_full_doc_20260215_123349/` | Stage 2: MAP | **`checkpoint-7000`** | ROUGE-L: 0.189 (word), Semantic: 0.754 |
| `banglaT5_reduce_task_20260216_222141/` | Stage 3: Reduce (attempt 1) | `checkpoint-500` | Early attempt |
| `banglaT5_reduce_task_20260217_111025/` | Stage 3: Reduce (attempt 2) | **`checkpoint-6000`** | Used in final pipeline |

Each model directory contains:
- `checkpoint-*/` — Model checkpoints (model.safetensors, optimizer.pt, scheduler.pt, tokenizer, trainer_state)
- `final_model/` — Best/final model (when available)
- `runs/` — TensorBoard logs
- `logs/` — Training logs
- `training_config.json` — Full hyperparameter record
- `all_results.json` — Final evaluation results
- `eval_results.json` — Validation results
- `test_results.json` — Test results
- `train_results.json` — Training results

### Result Files

| File | Content |
|------|---------|
| `pipeline_eval_results.json` | Full pipeline evaluation on 1,985 test articles |
| `pipeline_eval_results_OLD.json` | Previous pipeline evaluation results |
| `checkpoint_evaluation_results.json` | All MAP model checkpoints evaluated |
| `pipeline_output.txt` | Detailed per-article pipeline output |
| `sample_predictions_output.txt` | 10 sample predictions for visual inspection |

---

## 10. Key Design Decisions & Lessons Learned

### Decision 1: Word-Level vs Character-Level ROUGE

**Problem**: Bangla is morphologically rich. Character-level ROUGE inflates scores (~0.46) compared to word-level (~0.19).

**Decision**: Later scripts (full doc, reduce, pipeline) use word-level ROUGE for honest, comparable metrics. Earlier scripts (teacher, chunked) used character-level.

**Lesson**: Always specify ROUGE tokenization when reporting scores.

### Decision 2: Zero Label Smoothing for Full Document Model

**Decision**: `label_smoothing = 0.0` for the MAP and REDUCE models.

**Rationale**: Hard targets force the model to exactly reproduce gold summaries. This teaches precise salience extraction. The teacher model (Stage 0) used 0.1 label smoothing, which was suitable for general compression but not for learning importance ranking.

### Decision 3: No Evaluation During Full Document Training

**Decision**: `eval_strategy = "no"`, save ALL checkpoints.

**Rationale**: Evaluation with generation is slow on large datasets. Instead, save all checkpoints and evaluate the best ones post-training using `check_point_checker.py`. This proved efficient — checkpoint-7000 was identified as best across all metrics.

### Decision 4: Transfer Learning for Reduce Model

**Decision**: Initialize REDUCE model from MAP checkpoint-7000, not from pretrained HuggingFace model.

**Rationale**: The REDUCE model benefits from the MAP model's knowledge of Bangla summarization. Starting from scratch would require relearning language + domain + task, while transfer learning only requires learning the merge task.

### Decision 5: Teacher-Generated Reduce Data

**Decision**: Use the MAP model (checkpoint-7000) to generate chunk summaries for REDUCE training data, not use gold summary fragments.

**Rationale**: This is the key insight about train/inference distribution matching. Gold fragments are clean and well-written. Real chunk summaries are noisy, redundant, and stylistically inconsistent. Training on teacher-generated data teaches:
- Deduplication
- Contradiction resolution
- Missing information tolerance
- Order invariance

### Decision 6: Disabling Solution 4 (Attention Bias)

**Decision**: `USE_ATTENTION_BIAS = False` in all production models.

**Rationale**: Small models (296M parameters) get confused by multiple architectural signals. The simpler Solutions 1-3 were sufficient. Attention bias adds complexity without clear benefit at this model scale.

### Decision 7: Slow BanglaT5 Tokenizer

**Decision**: Use `T5Tokenizer(use_fast=False)` instead of fast tokenizer.

**Rationale**: The fast tokenizer (tiktoken-based) has issues with Bangla text encoding/decoding. The slow SentencePiece tokenizer handles Bangla correctly. The speed difference is negligible compared to model inference time.

### Lesson: The ROUGE Trap

The biggest lesson: **high ROUGE scores ≠ high quality**. The teacher model had ROUGE-L 0.465 but produced mediocre summaries. The full document model had lower ROUGE-L (0.189 word-level) but much better semantic quality (BERTScore 0.731, Semantic Similarity 0.754). Always use multiple metrics including semantic ones.

---

## 11. Utility Scripts Reference

### `bangla_sentence_splitter.py` — Core Sentence Segmenter

**22 Bangla abbreviations** that are NOT treated as sentence boundaries:
`ড.`, `ডা.`, `প্রফ.`, `মো.`, `মোঃ`, `জনাব`, `বেগম`, `সৈয়দ`, `হাজী`, `আলহাজ্ব`, `মি.`, `মিঃ`, `মিসেস`, `তৎ`, `নং`, `পৃ.`, `ই.`, `ড`, `চা.`, `কো.`, `রাজ.`, `উৎ`

**Sentence terminators**: `।` (Danda), `॥` (Double Danda), `?`, `!`, and English `.` followed by a space and uppercase/Bangla script.

### `chunk_processor.py` — Chunking Algorithm

**Algorithm pseudocode**:
```
1. Split article into sentences
2. Pre-compute BPE token count for each sentence (batched)
3. current_chunk = []
4. For each sentence:
   a. If adding sentence would exceed 900 tokens:
      - Finalize current chunk
      - Start new chunk with last 3 sentences (overlap)
   b. If single sentence > 900 tokens:
      - Add as standalone chunk (unavoidable truncation)
   c. Otherwise: add sentence to current chunk
5. If last chunk < 50 tokens and previous chunk exists:
   - Merge with previous chunk (if combined ≤ 110% of max)
   - Otherwise: keep as separate chunk
6. Update total_chunks count on all chunks
```

### `check_setup.py` — Pre-Training Validation

Checks before training:
1. Required files exist and have content
2. Dataset format is valid JSON with `text` and `summary` fields
3. GPU is available with sufficient memory (warns < 8GB)
4. All required Python packages are installed
5. Disk space is sufficient (warns < 10GB)

---

## 12. Configuration Reference

### All Training Configurations Compared

| Parameter | Teacher (Stage 0) | Chunked (Stage 1) | Full Doc (Stage 2) | Reduce (Stage 3) |
|-----------|-------------------|-------------------|-------------------|-------------------|
| **Script** | `train_bangla_teacher.py` | `train_bangla_chunked.py` | `train_full_document.py` | `train_reduce_task.py` |
| **Base Model** | `csebuetnlp/banglaT5` | `csebuetnlp/banglaT5` | `csebuetnlp/banglaT5` | MAP checkpoint-7000 |
| **Input Prefix** | `"summarize: "` | `"summarize bangla news: "` | `"summarize bangla news: "` | `"summarize multiple summaries: "` |
| **Data** | Short articles only | Pre-split JSONL | Pre-split JSON | Reduce data |
| **Train Samples** | 5,972 | 1,888 | ~63,602 | Variable |
| **Epochs** | 15 | 25 | 15 | 15 |
| **Learning Rate** | 5e-5 | 3e-5 | 3e-5 | 3e-5 |
| **Effective Batch** | 32 | 32 | 32 | 32 |
| **Max Target Len** | 192 | 256 | 256 | 256 |
| **Label Smoothing** | 0.1 | 0.05 | 0.0 | 0.0 |
| **Beams** | 4 | 6 | 5 | N/A |
| **Rep. Penalty** | 1.2 | 1.15 | 1.0 | N/A |
| **Gradient Ckpt** | OFF | OFF | ON | ON |
| **Eval Strategy** | Loss only | ROUGE w/ gen | None | None |
| **Best Metric** | `eval_loss` | `eval_rougeL` | N/A | N/A |
| **ROUGE Type** | Character | Character | Word | Word |
| **Bangla Char Fix** | No | No | Yes | Yes |
| **CLI Args** | No | No | Yes | Yes |

### All Generation Configurations Compared

| Parameter | Teacher | Chunked | Full Doc | Pipeline MAP | Pipeline REDUCE | Pipeline Single |
|-----------|---------|---------|----------|-------------|----------------|-----------------|
| max_length | 192 | 256 | 256 | 256 | 256 | 256 |
| min_length | — | 64 | 64 | 40 | 80 | 60 |
| num_beams | 4 | 6 | 5 | 5 | 5 | 5 |
| length_penalty | 1.0 | 1.2 | 1.0 | 1.2 | 2.0 | 1.5 |
| no_repeat_ngram | 2 | 3 | 3 | 3 | 3 | 3 |
| rep_penalty | 1.2 | 1.15 | 1.0 | 1.0 | 1.0 | 1.0 |
| early_stopping | True | True | True | True | True | True |

---

## 13. Installation & Quick Start

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support (RTX 5080 or similar)
- ~50 GB disk space for models and data

### Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\Activate.ps1

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
# Core
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
accelerate>=0.24.0

# Evaluation
rouge-score>=0.1.2
bert-score>=0.3.13
scikit-learn>=1.3.0

# Data processing
numpy>=1.24.0
pandas>=2.0.0

# Monitoring
tensorboard>=2.14.0

# Utilities
tqdm>=4.66.0
sentencepiece>=0.1.99
sacrebleu>=2.3.0
```

### Quick Start: Run the Pipeline

```bash
# Summarize sample articles from the test set
python run_pipeline.py --test --num_samples 10

# Summarize custom text
python run_pipeline.py --text "আপনার বাংলা নিবন্ধ এখানে..."

# Summarize from file
python run_pipeline.py --file article.txt
```

### Quick Start: Evaluate the Pipeline

```bash
# Full test-set evaluation (batched, faster)
python eval_pipeline_batched.py

# Sequential evaluation (slower but resumable)
python eval_pipeline.py

# Evaluate a single checkpoint
python eval_checkpoint.py --checkpoint ./banglaT5_full_doc_20260215_123349/checkpoint-7000

# Compare two models
python compare_models.py --old_model ./model_a --new_model ./model_b --samples 20
```

### Quick Start: Train from Scratch

```bash
# Step 0: Verify environment
python check_setup.py

# Step 1: Split dataset (if not already done)
python split_dataset.py

# Step 2: Verify data integrity
python verify_data.py

# Step 3: Train MAP model (most important!)
python train_full_document.py --epochs 15

# Step 4: Evaluate MAP checkpoints
python check_point_checker.py

# Step 5: Generate reduce training data
python generate_reduce_data.py

# Step 6: Train REDUCE model
python train_reduce_task.py --model ./banglaT5_full_doc_*/checkpoint-7000

# Step 7: Evaluate reduce checkpoints
python eval_reduce_checkpoints.py

# Step 8: Run full pipeline evaluation
python eval_pipeline_batched.py
```

### Using the Summarizer in Your Code

```python
from run_pipeline import ChunkedSummarizer

# Initialize with model paths
summarizer = ChunkedSummarizer(
    map_model_path="./banglaT5_full_doc_20260215_123349/checkpoint-7000",
    reduce_model_path="./banglaT5_reduce_task_20260217_111025/checkpoint-6000"
)

# Summarize any Bangla text
article = "আপনার দীর্ঘ বাংলা নিবন্ধ এখানে..."
result = summarizer.summarize(article)

print(f"Summary: {result['summary']}")
print(f"Method: {result['method']}")        # single_pass / single_chunk / map_reduce
print(f"Chunks: {result['num_chunks']}")
print(f"Input tokens: {result['input_tokens']}")

# Access chunk-level summaries (for map_reduce method)
if result['chunk_summaries']:
    for i, cs in enumerate(result['chunk_summaries']):
        print(f"  Chunk {i+1}: {cs[:100]}...")
```

---

## 14. Known Issues & Solutions

### Issue 1: Corrupted Training Samples

**Problem**: Some samples in reduce data had repetitive text with `[CHUNK]` markers (one sample had 695 repetitions).  
**Detection**: Use `verify_data.py` and `_check_reduce.py`  
**Solution**: Quality filtering in `generate_reduce_data.py` removes samples with extreme chunk counts.

### Issue 2: Precomposed Bangla Characters → UNK Tokens

**Problem**: Characters like `য়`, `ড়`, `ঢ়` are precomposed Unicode forms that BanglaT5's SentencePiece vocabulary maps to `<unk>`.  
**Solution**: `fix_bangla_for_tokenizer()` decomposes them into base character + nukta (়).  
**Verification**: `verify_data.py` checks for remaining precomposed characters.

### Issue 3: Memory Issues on Consumer GPUs

**Problem**: Large batch sizes or multiple models loaded simultaneously cause OOM.  
**Solution**: 
- bf16 inference (`torch.bfloat16`) halves memory
- Gradient checkpointing during training
- `eval_pipeline_batched.py` frees MAP model before loading REDUCE model
- Effective batch size 32 via gradient accumulation (4 physical × 8 accumulation)

### Issue 4: Slow Tokenizer Required

**Problem**: Fast tokenizer (tiktoken-based) has issues with Bangla text.  
**Solution**: `load_tokenizer()` tries `T5Tokenizer(use_fast=False)` first, falls back to `AutoTokenizer`.

### Issue 5: Infinite Loop in Chunking

**Problem**: Certain articles (e.g., article 400) caused infinite loops in the chunker.  
**Solution**: `prepare_gt1000_training_data.py` has `SKIP_ARTICLES = [400]` and a safety limit of `max_iterations = len(sentences) * 100`.

### Issue 6: ROUGE Score Confusion (Char vs Word Level)

**Problem**: Different scripts report different ROUGE scales, making comparison misleading.  
**Solution**: This documentation clearly labels which tokenization each script uses. Always check the ROUGE tokenizer before comparing scores.

---

## 15. Research Contributions

1. **MapReduce for Bangla Summarization**: First implementation of a two-stage MapReduce architecture specifically for Bangla news summarization
2. **Teacher-Generated Reduce Data**: Using a trained teacher model to generate noisy-but-realistic chunk summaries for training the reduce model, with corruption augmentations (shuffle, drop, duplicate)
3. **Bangla Character Decomposition Fix**: Identifying and fixing the precomposed Unicode character issue in BanglaT5's SentencePiece vocabulary
4. **Sentence-Aligned Chunking for Bangla**: Rule-based Bangla sentence segmenter respecting 22 abbreviations, quoted text, and decimal numbers
5. **Comprehensive Multi-Metric Evaluation**: Combining ROUGE, BERTScore, Semantic Similarity, and BLEU for holistic quality assessment
6. **Analysis of the "ROUGE Trap"**: Documenting how high character-level ROUGE scores can mask poor semantic quality in morphologically-rich languages

---

## 16. References

- **BanglaT5**: Bhattacharjee et al. (2022) — `csebuetnlp/banglaT5` on Hugging Face. 296M parameter T5 model pretrained on Bangla text.
- **XL-Sum Dataset**: Hasan et al. (2021) — Cross-lingual Abstractive Summarization for 44 Languages. Source of Bangla news articles from BBC Bangla.
- **T5 Architecture**: Raffel et al. (2020) — Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.
- **MapReduce Summarization**: Inspired by distributed computing MapReduce paradigm applied to hierarchical text summarization.
- **Sentence-Transformers**: Reimers & Gurevych (2019) — Sentence-BERT for semantic similarity evaluation.
- **BERTScore**: Zhang et al. (2020) — BERTScore: Evaluating Text Generation with BERT.

---

## Project Timeline

| Date | Milestone |
|------|-----------|
| Feb 10, 2026 | Stage 0: Production/teacher model trained (`banglaT5_production_20260210_131619`) |
| Feb 13, 2026 | Stage 1: Chunked model trained (`banglaT5_chunked_20260213_193538`) |
| Feb 14, 2026 | Quality analysis — identified ROUGE trap and salience learning gap |
| Feb 15, 2026 | Stage 2: Full document model trained (`banglaT5_full_doc_20260215_123349`) |
| Feb 16, 2026 | Stage 3: Reduce model attempt 1 (`banglaT5_reduce_task_20260216_222141`) |
| Feb 17, 2026 | Stage 3: Reduce model attempt 2 with teacher-generated data (`banglaT5_reduce_task_20260217_111025`) |
| Feb–Mar 2026 | Comprehensive evaluation and documentation |

---

**License**: This project uses the BanglaT5 model which is under the MIT License.  
**Hardware**: NVIDIA RTX 5080 (16 GB VRAM)  
**Contact**: For questions or collaboration, review the code starting with `run_pipeline.py`.
