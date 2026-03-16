# Multilingual Summarization via Knowledge Distillation

> **Compressing a 966M-parameter multilingual teacher into a 300M-parameter student while retaining 78–89% of summarization quality across 5 languages.**

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Architecture & Pipeline](#architecture--pipeline)  
3. [Models](#models)  
4. [Dataset](#dataset)  
5. [Training Pipeline](#training-pipeline)  
6. [Evaluation Metrics](#evaluation-metrics)  
7. [Results](#results)  
8. [File Structure](#file-structure)  
9. [Setup & Installation](#setup--installation)  
10. [Usage Guide](#usage-guide)  
11. [Troubleshooting](#troubleshooting)  
12. [Design Decisions & Lessons Learned](#design-decisions--lessons-learned)  

---

## Project Overview

This project implements **knowledge distillation** for multilingual text summarization. A large **teacher model** (`csebuetnlp/mT5_multilingual_XLSum`, 966M parameters) is fine-tuned per language, then its knowledge is transferred to a smaller **student model** (`google/mt5-small`, 300M parameters) through **offline pseudo-labeling**.

### Key Highlights

- **5 Languages**: Hindi, Urdu, Russian, Portuguese, Persian  
- **Teacher**: `csebuetnlp/mT5_multilingual_XLSum` (966.6M params)  
- **Student**: `google/mt5-small` (300M params) — **3.2× compression**  
- **Method**: Offline pseudo-labeling (teacher generates summaries once, student trains on cached predictions)  
- **Speed**: 100× faster than online distillation (0.16s/iter vs 16s/iter)  
- **Retention**: Student retains 78–89% of teacher ROUGE-L across languages  

### Motivation

Large language models produce excellent summaries but are expensive to deploy. Knowledge distillation transfers the teacher's summarization ability into a smaller model that is:

- **3.2× smaller** in memory  
- **Faster at inference** (fewer parameters)  
- **Cheaper to serve** in production  

---

## Architecture & Pipeline

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌───────────────┐    ┌─────────────────┐  │
│  │  XLSum   │───▶│  Preprocess   │───▶│  Language CSVs  │  │
│  │  Dataset  │    │ Filter ≤512   │    │  (train/val/    │  │
│  │ (300K+)  │    │   tokens      │    │   test splits)  │  │
│  └──────────┘    └───────────────┘    └────────┬────────┘  │
│                                                 │           │
│                                    ┌────────────▼────────┐  │
│                                    │  Fine-tune Teacher  │  │
│                                    │  mT5_XLSum (966M)   │  │
│                                    │  10 epochs, LR=2e-5 │  │
│                                    └────────────┬────────┘  │
│                                                 │           │
│                                    ┌────────────▼────────┐  │
│                                    │ Generate Pseudo-    │  │
│                                    │ Labels (beam=6)     │  │
│                                    │ Teacher predicts    │  │
│                                    │ once, save to CSV   │  │
│                                    └────────────┬────────┘  │
│                                                 │           │
│                                    ┌────────────▼────────┐  │
│                                    │  Train Student      │  │
│                                    │  mt5-small (300M)   │  │
│                                    │  3 epochs, LR=5e-5  │  │
│                                    │  On cached labels   │  │
│                                    └─────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Knowledge Distillation Strategy

We use **offline pseudo-labeling** rather than online distillation:

```
┌─────────────────────────────────────────────────────────────────┐
│          ONLINE (Slow)              OFFLINE (Fast - Used)       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  For each batch:                   Phase 1 (one-time):          │
│  ┌─────────┐  ┌─────────┐         ┌─────────┐  ┌──────────┐   │
│  │ Teacher │  │ Student │         │ Teacher │──▶│ Pseudo-  │   │
│  │ (966M)  │  │ (300M)  │         │ (966M)  │  │ Labels   │   │
│  │ forward │  │ forward │         │ forward │  │ (CSV)    │   │
│  │ pass    │  │ pass    │         │ only    │  └──────────┘   │
│  └────┬────┘  └────┬────┘         └─────────┘                  │
│       │            │                                            │
│       ▼            ▼               Phase 2 (fast):              │
│  ┌─────────────────────┐          ┌──────────┐  ┌─────────┐   │
│  │   KL Divergence     │          │ Pseudo-  │──▶│ Student │   │
│  │   + Hard Loss       │          │ Labels   │  │ (300M)  │   │
│  │   Both models in    │          │ (CSV)    │  │ only    │   │
│  │   GPU memory        │          └──────────┘  └─────────┘   │
│  └─────────────────────┘                                       │
│                                                                 │
│  Speed: 16s/iteration            Speed: 0.16s/iteration        │
│  GPU: Teacher + Student          GPU: Student only              │
│  Memory: ~8GB                    Memory: ~3GB                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Why offline?** Loading the 966M teacher for every training batch is extremely slow. By generating pseudo-labels once and caching them, the student trains 100× faster with only itself in GPU memory.

---

## Models

### Teacher: `csebuetnlp/mT5_multilingual_XLSum`

| Property | Value |
|---|---|
| Architecture | mT5 (Encoder-Decoder Transformer) |
| Parameters | 966.6M |
| Pre-training | mT5 base → fine-tuned on XLSum (45 languages) |
| Vocabulary | 250,112 tokens (SentencePiece) |
| Input format | `"summarize: " + article` |
| Max input length | 512 tokens |
| Max output length | 256 tokens |

### Student: `google/mt5-small`

| Property | Value |
|---|---|
| Architecture | mT5 (Encoder-Decoder Transformer) |
| Parameters | 300M (estimated from mt5-small) |
| Pre-training | mT5-small (multilingual C4) |
| Vocabulary | 250,112 tokens (same SentencePiece as teacher) |
| Compression ratio | 3.2× smaller than teacher |
| Input format | `"summarize: " + article` (matches teacher) |

### Why these models?

- **Same tokenizer**: Both use the mT5 SentencePiece vocabulary, so pseudo-labels transfer perfectly  
- **Same architecture family**: Both are encoder-decoder transformers, enabling direct output matching  
- **Multilingual**: Both support 100+ languages natively  
- **Pre-trained on summarization**: The teacher was already fine-tuned on XLSum, giving a strong starting point  

---

## Dataset

### Source: XLSum

[XLSum](https://huggingface.co/datasets/csebuetnlp/xlsum) is a large-scale multilingual summarization dataset from BBC articles, covering 44 languages.

### Preprocessing Pipeline

```
XLSum Full Dataset (300K+ articles)
         │
         ▼
┌────────────────────────┐
│  Filter by language    │  → Keep only target 5 languages
│  Filter ≤512 tokens    │  → Remove long articles (mT5 tokenizer)
│  Shuffle (seed=42)     │  → Reproducible randomization
│  Split 80/10/10        │  → Train / Validation / Test
└────────────────────────┘
         │
         ▼
   Language CSVs (text, summary)
```

### Dataset Statistics

| Language | Total in XLSum | After ≤512 Filter | Train | Val | Test |
|---|---|---|---|---|---|
| Hindi | 70,778 | 16,404 (23.2%) | 13,123 | 1,640 | 1,641 |
| Urdu | 67,665 | 18,600 (27.5%) | 14,880 | 1,860 | 1,860 |
| Russian | 62,243 | 19,824 (31.8%) | 15,859 | 1,982 | 1,983 |
| Portuguese | 57,402 | 17,640 (30.7%) | 14,112 | 1,764 | 1,764 |
| Persian | 47,251 | 9,653 (20.4%) | 7,722 | 965 | 966 |

### CSV Format

Each CSV contains two columns:

| Column | Description |
|---|---|
| `text` | Full article text |
| `summary` | Reference summary |

After pseudo-label generation, a third column is added:

| Column | Description |
|---|---|
| `teacher_summary` | Teacher model's generated summary |

---

## Training Pipeline

### Step 1: Preprocess Dataset

```bash
python preprocess_multilingual.py
```

Filters XLSum for target languages and token limits. Creates `./preprocessed_data/{language}/train.csv`, `val.csv`, `test.csv`.

For Russian, Portuguese, Persian:
```bash
python count_and_create_datasets.py
```

### Step 2: Fine-tune Teacher

```bash
python setup_language.py hindi    # Configure for a language
python train_teacher.py           # Train teacher model
```

**Teacher Training Hyperparameters:**

| Parameter | Value |
|---|---|
| Base model | `csebuetnlp/mT5_multilingual_XLSum` |
| Learning rate | 2e-5 |
| Batch size | 8 |
| Gradient accumulation | 4 (effective batch = 32) |
| Epochs | 10 |
| Warmup steps | 500 |
| Weight decay | 0.01 |
| Precision | bf16 |
| Beam search (eval) | 6 beams |
| Early stopping | Patience 5 on val ROUGE-L |
| Optimizer | AdamW |

Output: `./teachers/{language}_teacher_{timestamp}/final_model/`

### Step 3: Generate Pseudo-Labels

```bash
python generate_teacher_labels.py
```

Runs the fine-tuned teacher on all train/val/test splits (inference only, no gradients). Saves CSVs with `teacher_summary` column to `./preprocessed_data/{language}_finetuned_teacher_labels/`.

**Generation config:** beam search with 6 beams, max 256 output tokens.

### Step 4: Train Student (Fast)

```bash
python train_student_fast.py
```

Trains the student on cached teacher predictions. No teacher model in memory.

**Student Training Hyperparameters:**

| Parameter | Value |
|---|---|
| Base model | `google/mt5-small` |
| Learning rate | 5e-5 |
| Batch size | 32 |
| Gradient accumulation | 1 |
| Epochs | 3 |
| Warmup steps | 500 |
| Weight decay | 0.01 |
| Precision | bf16 |
| Training target | `teacher_summary` (pseudo-labels) |

Output: `./students/{language}_student_fast_{timestamp}/`

### Automated Pipeline

For training multiple languages sequentially:

```bash
python train_remaining_languages.py
```

This script:
1. Updates config files for each language
2. Trains teacher (or reuses existing)
3. Generates pseudo-labels using the correct teacher
4. Trains student on cached labels
5. Continues to next language on failure

---

## Evaluation Metrics

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

| Metric | Description |
|---|---|
| **ROUGE-1** | Unigram overlap between prediction and reference |
| **ROUGE-2** | Bigram overlap between prediction and reference |
| **ROUGE-L** | Longest Common Subsequence (LCS) based metric |

Uses space-based tokenization (word-level splitting) — works well for all target languages since they all use spaces between words.

### BLEU (Bilingual Evaluation Understudy)

Measures n-gram precision with brevity penalty. Computed using SacreBLEU for reproducibility. Scored at corpus level.

### BERTScore

Uses contextual embeddings from `bert-base-multilingual-cased` to compute semantic similarity between predictions and references. Reports Precision, Recall, and F1.

### Semantic Similarity

Uses `paraphrase-multilingual-MiniLM-L12-v2` sentence transformer to encode predictions and references, then computes pairwise cosine similarity.

### Retention Percentage

```
Retention % = (Student ROUGE-L / Teacher ROUGE-L) × 100
```

Measures how much of the teacher's quality the student preserves after distillation.

---

## Results

### Teacher Performance (Fine-tuned)

| Language | ROUGE-1 | ROUGE-2 | ROUGE-L | Train Samples |
|---|---|---|---|---|
| Hindi | 0.4191 | 0.2175 | 0.3715 | 13,123 |
| Urdu | 0.4183 | 0.2128 | 0.3717 | 14,880 |
| Russian | 0.2444 | 0.1068 | 0.2199 | 15,859 |
| Portuguese | 0.3217 | 0.1372 | 0.2891 | 14,112 |
| Persian | 0.3522 | 0.1671 | 0.3244 | 7,722 |

> **Note**: Russian's lower teacher performance is because the base model (`mT5_multilingual_XLSum`) was already well-optimized for Russian during its original training; further fine-tuning on a filtered subset provided minimal improvement.

### Student Performance (Distilled)

| Language | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-L Retention |
|---|---|---|---|---|
| Hindi | 0.3441 | 0.1647 | 0.3083 | 83.0% |
| Urdu | 0.3702 | 0.1946 | 0.3299 | 88.8% |
| Russian | 0.2016 | 0.0959 | 0.1882 | 85.6% |
| Portuguese | 0.2860 | 0.1223 | 0.2523 | 87.3% |
| Persian | 0.2915 | 0.1298 | 0.2538 | 78.2% |

### Additional Metrics (BLEU, BERTScore, Semantic Similarity)

> See `comprehensive_results.json` for complete per-language breakdown including BLEU, BERTScore (P/R/F1), and semantic similarity scores.

### Retention Analysis

```
Teacher vs Student ROUGE-L Retention

Hindi       ████████████████████████████████████████░░░░░░░░  83.0%
Urdu        ████████████████████████████████████████████░░░░  88.8%
Russian     █████████████████████████████████████████░░░░░░░  85.6%
Portuguese  ████████████████████████████████████████████░░░░  87.3%
Persian     ███████████████████████████████████████░░░░░░░░░  78.2%

Average Retention: 84.6%
```

---

## File Structure

```
multilingual_support/
│
├── 📄 README.md                          # This file
├── 📄 comprehensive_results.json         # All evaluation metrics
│
├── 📊 Data & Preprocessing
│   ├── xlsum_all_train.csv               # Raw XLSum dataset (all languages)
│   ├── preprocess_multilingual.py        # Filter for South Asian languages
│   ├── count_and_create_datasets.py      # Filter for Russian/Portuguese/Persian
│   └── preprocessed_data/
│       ├── hindi/                        # train.csv, val.csv, test.csv
│       ├── urdu/
│       ├── russian/
│       ├── portuguese/
│       ├── persian/
│       ├── hindi_finetuned_teacher_labels/    # + teacher_summary column
│       ├── urdu_finetuned_teacher_labels/
│       ├── russian_finetuned_teacher_labels/
│       ├── portuguese_finetuned_teacher_labels/
│       └── persian_finetuned_teacher_labels/
│
├── 🎓 Teacher Training
│   ├── train_teacher.py                  # Fine-tune mT5_XLSum teacher
│   ├── setup_language.py                 # Configure for specific language
│   └── teachers/
│       ├── hindi_teacher_20260306_*/     # Fine-tuned teacher models
│       ├── urdu_teacher_20260306_*/
│       ├── russian_teacher_20260307_*/
│       ├── portuguese_teacher_20260307_*/
│       └── persian_teacher_20260307_*/
│
├── 🧑‍🎓 Student Training (Knowledge Distillation)
│   ├── generate_teacher_labels.py        # Offline pseudo-label generation
│   ├── train_student_fast.py             # Fast student training (offline KD)
│   ├── train_student.py                  # Online KD (deprecated - too slow)
│   └── students/
│       ├── hindi_student_fast_*/         # Distilled student models
│       ├── urdu_student_fast_*/
│       ├── russian_student_fast_*/
│       ├── portuguese_student_fast_*/
│       └── persian_student_fast_*/
│
├── 🔧 Automation & Utilities
│   ├── train_remaining_languages.py      # Automated multi-language pipeline
│   ├── smoke_test.py                     # Quick pipeline validation
│   └── evaluate_all_models.py            # Comprehensive evaluation script
│
└── 📋 Legacy / Intermediate
    ├── training_results.json             # Early failed run log
    └── train_all_languages.py            # Earlier automation attempt
```

### Each Teacher Directory Contains:

```
teachers/{language}_teacher_{timestamp}/
├── final_model/                # Complete model (weights, config, tokenizer)
│   ├── model.safetensors       # Model weights
│   ├── config.json             # Model architecture config
│   ├── tokenizer.json          # Tokenizer
│   ├── generation_config.json  # Beam search settings
│   └── ...
├── checkpoint-{step}/          # Training checkpoints
├── training_config.json        # Hyperparameters used
├── test_results.json           # Test set ROUGE scores
├── eval_results.json           # Validation set metrics
├── train_results.json          # Training metrics
├── all_results.json            # Combined metrics
└── logs/                       # TensorBoard logs
```

### Each Student Directory Contains:

```
students/{language}_student_fast_{timestamp}/
├── model.safetensors           # Student model weights
├── config.json                 # Model config
├── tokenizer.json              # Tokenizer (same as teacher)
├── generation_config.json      # Generation settings
├── test_results.json           # Test set evaluation
├── training_args.bin           # Serialized training arguments
└── checkpoint-{step}/          # Best checkpoint
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (tested on NVIDIA GPU with bf16 support)
- ~10GB GPU memory for teacher training
- ~4GB GPU memory for student training

### Install Dependencies

```bash
pip install torch transformers datasets sentencepiece protobuf
pip install rouge-score sacrebleu bert-score sentence-transformers
pip install pandas numpy tqdm scikit-learn
```

### Verify GPU

```python
import torch
print(torch.cuda.is_available())       # Should be True
print(torch.cuda.get_device_name(0))   # GPU name
```

---

## Usage Guide

### Quick Start: Train One Language

```bash
# 1. Preprocess
python preprocess_multilingual.py

# 2. Configure for Hindi
python setup_language.py hindi

# 3. Train teacher (~50 min)
python train_teacher.py

# 4. Generate pseudo-labels (~25 min)
python generate_teacher_labels.py

# 5. Train student (~5 min)
python train_student_fast.py
```

### Train All Languages Automatically

```bash
# Edit LANGUAGES list in train_remaining_languages.py
# Add any existing teachers to EXISTING_TEACHERS dict
python train_remaining_languages.py
```

### Validate Pipeline (Smoke Test)

```bash
# Quick 3-minute test on 100 Hindi samples
python smoke_test.py
```

### Run Comprehensive Evaluation

```bash
# Computes ROUGE, BLEU, BERTScore, Semantic Similarity for all models
python evaluate_all_models.py
```

### Use a Trained Student for Inference

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_path = "./students/hindi_student_fast_20260306_212437"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to("cuda")

article = "summarize: " + "Your article text here..."
inputs = tokenizer(article, max_length=512, truncation=True, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=256, num_beams=6)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(summary)
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|---|---|
| `ModuleNotFoundError: sentencepiece` | `pip install sentencepiece protobuf` |
| `Seq2SeqTrainer: unexpected tokenizer` | Remove `tokenizer=` parameter from Trainer |
| CUDA out of memory | Reduce `BATCH_SIZE` or use gradient accumulation |
| Windows path `SyntaxError` | Use forward slashes `/` in paths |
| Poor ROUGE (< 0.10) | Check `USE_PREFIX` matches teacher training |
| Slow training (>10s/iter) | Use offline distillation (`train_student_fast.py`) |
| `GenerationConfig` error | Use `generation_config=GenerationConfig.from_pretrained()` |

### Task Prefix

The mT5_multilingual_XLSum model expects inputs prefixed with `"summarize: "`. If pseudo-labels are generated **without** the prefix but the teacher was trained **with** it, outputs will be garbage.

**Rule**: `USE_PREFIX` must be consistent between:
1. Teacher training (`train_teacher.py` — always True)
2. Pseudo-label generation (`generate_teacher_labels.py`)
3. Student training (`train_student_fast.py`)

---

## Design Decisions & Lessons Learned

### Why Offline Distillation?

**Online distillation** (loading both teacher and student in GPU memory) was tested first:
- Speed: **16 seconds/iteration** — both models do forward passes every batch
- Memory: ~8GB (teacher 966M + student 300M)

**Offline pseudo-labeling** replaced it:
- Speed: **0.16 seconds/iteration** — student trains alone on cached labels
- Memory: ~3GB (student only during training)
- Trade-off: Slightly less flexible (can't do temperature-based soft label distillation)
- Result: 100× speedup with comparable quality

### Why Not Use the Base Model Directly?

For Hindi and Urdu, fine-tuning improved ROUGE-L by ~0.10 over the base model. For Russian, the base model was already well-optimized (pre-trained on full XLSum), so fine-tuning gave minimal gains.

**Recommendation**: For languages where the base model was trained on the full XLSum dataset, skip fine-tuning and use the base model directly as teacher.

### Dataset Size Impact

| Train Samples | Example | Teacher ROUGE-L | Notes |
|---|---|---|---|
| < 1,000 | Gujarati (799) | 0.225 | Insufficient for good fine-tuning |
| 7,000+ | Persian (7,722) | 0.324 | Decent performance |
| 13,000+ | Hindi (13,123) | 0.372 | Strong performance |
| 15,000+ | Russian (15,859) | 0.220 | Already optimized in base model |

### Token Filtering Trade-off

Filtering articles to ≤512 tokens ensures no truncation during training but reduces dataset size significantly (only 11–32% of articles pass). For low-resource languages, consider:
- Increasing max tokens to 768 or 1024
- Using truncation instead of filtering
- Augmenting with related language data

---

## Citation

If you use this codebase, please cite:

```bibtex
@misc{multilingual_kd_summarization_2026,
  title={Multilingual Summarization via Knowledge Distillation},
  year={2026},
  note={mT5_multilingual_XLSum teacher distilled to mt5-small student}
}
```

### Underlying Models & Dataset

```bibtex
@inproceedings{hasan2021xlsum,
  title={XL-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages},
  author={Hasan, Tahmid and others},
  booktitle={Findings of ACL},
  year={2021}
}

@article{xue2021mt5,
  title={mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer},
  author={Xue, Linting and others},
  journal={NAACL},
  year={2021}
}
```
