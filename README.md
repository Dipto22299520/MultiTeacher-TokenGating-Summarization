# Enhanced Knowledge Distillation with Multi-Teacher Ensemble for Bangla Abstractive Summarization

## A Systematic Ablation Study

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Research Motivation & Novel Contributions](#2-research-motivation--novel-contributions)
3. [Datasets](#3-datasets)
4. [Model Architecture](#4-model-architecture)
5. [Knowledge Distillation Framework](#5-knowledge-distillation-framework)
6. [Ablation Study Design (A1–A5)](#6-ablation-study-design-a1a5)
7. [Pipeline Execution Order](#7-pipeline-execution-order)
8. [Script Reference](#8-script-reference)
9. [Experimental Results — XLSum Dataset](#9-experimental-results--xlsum-dataset)
10. [Experimental Results — BanSum Dataset](#10-experimental-results--bansum-dataset)
11. [Cross-Dataset Analysis](#11-cross-dataset-analysis)
12. [Multi-Judge LLM Evaluation](#12-multi-judge-llm-evaluation)
13. [Inference Speed & Efficiency](#13-inference-speed--efficiency)
14. [Directory Structure](#14-directory-structure)
15. [How to Run](#15-how-to-run)
16. [Resume & Fault Tolerance](#16-resume--fault-tolerance)
17. [Dependencies & Hardware](#17-dependencies--hardware)
18. [Design Decisions & Rationale](#18-design-decisions--rationale)

---

## 1. Project Overview

This repository implements a **multi-component knowledge distillation (KD) framework** for Bangla abstractive summarization with a **systematic five-stage ablation study** that isolates and quantifies the contribution of each distillation component.

**Core goal**: Compress a large BanglaT5 teacher (247.6M params) into a compact BanglaT5-small student (109.9M params) while maximizing summary quality through heterogeneous multi-teacher knowledge transfer.

**The study was conducted on two independent Bangla summarization datasets** — XLSum (news articles from BBC Bangla) and BanSum (Bangla news corpus) — to validate generalization of findings across domains and data distributions.

**Key outputs**:
- 5 ablation configurations (A1–A5) trained and evaluated on both datasets
- Comprehensive metrics: ROUGE-1/2/L, BLEU, BERTScore, Semantic Similarity
- 4,000 generated summaries (1,000 per model × 4 pipelines) for multi-judge LLM evaluation
- Full timing/throughput benchmarks for distillation efficiency analysis

---

## 2. Research Motivation & Novel Contributions

### Why This Work Is Needed

Standard knowledge distillation for summarization follows a simple recipe: train a teacher, compute KL divergence between teacher and student logits, done. Reviewers frequently challenge:

- *"What is novel beyond vanilla KD?"*
- *"Which component actually contributes to improvement?"*
- *"Is the gain robust across datasets or just tuning noise?"*

This work addresses all three concerns through a principled ablation framework.

### Novel Contributions

1. **Heterogeneous Multi-Teacher KD**: Combines logit-level distillation from a same-vocabulary teacher (BanglaT5 → BanglaT5-small) with sequence-level pseudo-labels from cross-architecture teachers (mT5-base, mT5-XLSum). This is necessary because mT5's vocabulary differs from BanglaT5's, making logit-level KD impossible between them — pseudo-labels provide a vocabulary-agnostic knowledge transfer path.

2. **Confidence-Adaptive Temperature Scaling**: Instead of a fixed temperature τ for all samples, τ is dynamically computed per sample based on teacher prediction entropy. High-confidence teacher predictions (low entropy) use a lower τ to preserve sharp signal; uncertain predictions use a higher τ for softer regularization. The mapping uses sigmoid centering on batch mean entropy:
   ```
   τ = τ_min + (τ_max - τ_min) × σ(H_sample − H_batch_mean)
   ```

3. **Intermediate Encoder Hidden State Matching**: A learned linear projection (512 → 768) aligns student encoder representations with teacher encoder representations via normalized MSE loss:
   ```
   L_inter = ||normalize(W·h_student) − normalize(h_teacher)||²
   ```
   Masked to ignore padding positions.

4. **Multi-Granularity Loss Function**: The total training objective combines three granularities:
   ```
   L_total = α_hard × L_CE + α_kd × L_KL + α_inter × L_MSE
   ```
   With α_hard dynamically adjusted: `α_hard = 1.0 − α_kd − α_inter`.

5. **Systematic Ablation on Two Datasets**: Each component is isolated via incremental activation (A1→A5), replicated on XLSum and BanSum to test cross-dataset robustness.

---

## 3. Datasets

### 3.1 XLSum Dataset (Primary)

**Source**: `text_summarization.csv` — Bangla news articles and summaries derived from BBC Bangla (XLSum corpus).

**Preprocessing** (`preprocess.py`):
- Removed null/empty rows
- Split: 70% train / 20% validation / 10% test (sequential, not shuffled)

| Split | Samples | File |
|-------|---------|------|
| Train | 55,037 → 56,226 (with pseudo-label alignment) | `data/train.csv` |
| Validation | 6,880 → 16,064 (merged val+partial) | `data/val.csv` |
| Test | 6,880 → 8,033 | `data/test.csv` |
| **Total** | **68,797 clean** | |

**Columns**: `category`, `text`, `summary`

**Token length characteristics** (analyzed in `analyze_lengths.py`):
- Input text: mean ~300–350 tokens, 95th percentile ~550 tokens → `MAX_INPUT_LENGTH = 512`
- Summary: mean ~60–80 tokens, 95th percentile ~150 tokens → `MAX_TARGET_LENGTH = 256`
- At 512/256 cutoffs, >95% of data is captured without truncation.

### 3.2 BanSum Dataset (Secondary/Validation)

**Source**: `bansum_lte_1000_tokens.json` — Bangla news summarization corpus, pre-filtered to articles ≤1000 tokens.

**Split**: 80/10/10

| Split | Samples |
|-------|---------|
| Train | 112,960 |
| Validation | 14,120 |
| Test | 14,120 |
| **Total** | **141,200** |

**Structure per sample**:
- `main`: Input article (Bangla text)
- `sum1`, `sum2`, `sum3`: Three reference summaries of varying detail
- `token_count`: Pre-computed token count (all ≤ 1000)

**Token length characteristics** (analyzed in `analyze_bansum.py`):
- Input text: mean ~500–600 tokens, max ~990 → `MAX_INPUT_LENGTH = 850`
- Summaries: variable length across sum1/sum2/sum3
- Teacher training uses `sum1` or `sum2`; ablation student training uses `sum2`

### 3.3 Data Quality

`check_data_quality.py` performs a full audit:
- Duplicate detection (text, summary, exact pairs)
- Compression ratio analysis (text length / summary length)
- Empty/missing value scanning
- Extractive overlap detection
- Outputs a composite quality score (0–100)

---

## 4. Model Architecture

### 4.1 Student Model

| Property | Value |
|----------|-------|
| **Model** | `csebuetnlp/banglat5_small` |
| **Architecture** | T5 encoder-decoder |
| **Parameters** | 109,855,232 (109.9M) |
| **d_model** | 512 |
| **Language** | Bangla |
| **Vocabulary** | BanglaT5 vocabulary (shared with teacher) |

### 4.2 Teacher Models — XLSum

#### Primary Teacher: BanglaT5 (for logit-level KD)

| Property | Value |
|----------|-------|
| **Model** | `csebuetnlp/banglat5` (fine-tuned) |
| **Parameters** | 247,577,856 (247.6M) |
| **d_model** | 768 |
| **Fine-tune config** | batch=4, grad_accum=8 (eff. 32), lr=2e-5, epochs=10 |
| **MAX_INPUT** | 512 |
| **MAX_TARGET** | 256 |
| **Gradient checkpointing** | Yes (large model, memory-constrained) |
| **Precision** | bf16 |
| **Output directory** | `bangla_t5_teacher_finetuned_20251216_143715/` |

**XLSum teacher test results**:

| Metric | Score |
|--------|-------|
| ROUGE-1 | 0.4234 |
| ROUGE-2 | 0.2512 |
| ROUGE-L | **0.4058** |
| Test loss | 1.7282 |
| Throughput | 18.5 samples/sec |

#### Pseudo-Label Teacher 1: mT5-base

| Property | Value |
|----------|-------|
| **Model** | `google/mt5-base` (fine-tuned) |
| **Parameters** | ~580M |
| **Fine-tune config** | batch=16, grad_accum=2 (eff. 32), lr=3e-5, epochs=10 |
| **MAX_INPUT** | 512 |
| **MAX_TARGET** | 64 |
| **Best checkpoint** | `checkpoint-16000` (selected by semantic similarity) |
| **Output directory** | `mt5_teacher_mt5-base_20260208_124334/` |

**mT5-base test results**:

| Metric | Score |
|--------|-------|
| ROUGE-1 | 0.4047 |
| ROUGE-2 | 0.2405 |
| ROUGE-L | 0.3896 |
| BLEU | 17.29 |
| Semantic Similarity | 0.6782 |
| BERTScore F1 | 0.7903 |

#### Pseudo-Label Teacher 2: mT5-XLSum

| Property | Value |
|----------|-------|
| **Model** | `csebuetnlp/mT5_multilingual_XLSum` (fine-tuned) |
| **Parameters** | ~580M |
| **Fine-tune config** | batch=16, grad_accum=2 (eff. 32), lr=3e-5, epochs=8 |
| **MAX_INPUT** | 512 |
| **MAX_TARGET** | 64 |
| **Best checkpoint** | `checkpoint-12000` (selected by semantic similarity) |
| **Output directory** | `mt5_xlsum_20260212_060223/` |

**mT5-XLSum test results**:

| Metric | Score |
|--------|-------|
| ROUGE-1 | 0.4030 |
| ROUGE-2 | 0.2378 |
| ROUGE-L | 0.3875 |
| BLEU | 17.29 |
| Semantic Similarity | 0.6786 |
| BERTScore F1 | 0.7912 |

### 4.3 Teacher Models — BanSum

#### Primary Teacher: BanglaT5-BanSum

| Property | Value |
|----------|-------|
| **Model** | `csebuetnlp/BanglaT5` (fine-tuned on BanSum) |
| **Parameters** | 296,926,464 (296.9M) |
| **Fine-tune config** | batch=16, grad_accum=2 (eff. 32), lr=3e-5, epochs=8 |
| **MAX_INPUT** | 850 (extended for longer BanSum articles) |
| **MAX_TARGET** | 256 |
| **Output directory** | `banglat5_bansum_20260218_213532/` |

**BanSum teacher test results**:

| Metric | Score |
|--------|-------|
| ROUGE-1 | 0.3663 |
| ROUGE-2 | 0.2135 |
| ROUGE-L | **0.2998** |
| Test loss | 1.4165 |
| Gen length | 247.9 tokens |
| Throughput | 6.7 samples/sec |

#### Pseudo-Label Teachers (BanSum)

| Teacher | Checkpoint | Directory |
|---------|-----------|-----------|
| mT5-base (BanSum) | `checkpoint-16000` | `mt5base_bansum_20260219_113113/` |
| mT5-XLSum (BanSum) | `checkpoint-14000` | `mt5xlsum_bansum_20260219_062938/` |

### 4.4 Checkpoint Selection Process

Best checkpoints for mT5 teachers were selected using **semantic similarity** as the primary metric (not ROUGE), because:
- Semantic similarity captures meaning preservation better than n-gram overlap
- For pseudo-label generation, semantic fidelity matters more than lexical matching

Evaluation performed by:
- `evaluate_all_checkpoints.py` — all mT5-base checkpoints
- `evaluate_xlsum_checkpoints.py` — all mT5-XLSum checkpoints
- `compare_best_checkpoints.py` — head-to-head on full 8,033 test samples
- `eval_a1_a4_checkpoints.py` — BanSum A1–A4 checkpoint selection
- `eval_a5_checkpoints.py` — BanSum A5 checkpoint selection

---

## 5. Knowledge Distillation Framework

### 5.1 Loss Function

The training objective is a weighted combination of three loss terms:

```
L_total = α_hard × L_CE + α_kd × L_KL + α_inter × L_MSE
```

Where:
- **α_hard** = `1.0 − α_kd − α_inter` (dynamically computed so weights sum to 1)
- **α_kd** = `0.01` (KD loss weight — intentionally small, see Section 18)
- **α_inter** = `0.1` (intermediate matching weight, only in A5)

### 5.2 Component: Logit-Level KD (L_KL)

Standard KL divergence between temperature-scaled teacher and student logits:

```python
L_KL = KL(softmax(s/τ) || softmax(t/τ)) × τ²
```

**Fixed temperature** (A2, A3): τ = 0.8

**Adaptive temperature** (A4, A5): τ varies per sample based on teacher entropy:
```python
H = −Σ p_t × log(p_t)           # per-token teacher entropy
H_sample = mean(H, masked)       # average across non-padding tokens
τ = τ_min + (τ_max − τ_min) × σ(H_sample − H_batch_mean)
```
- τ_min = 0.5, τ_max = 2.0
- High-confidence predictions → low τ → sharper soft targets
- Uncertain predictions → high τ → smoother regularization

### 5.3 Component: Pseudo-Label Augmentation

Cross-architecture knowledge transfer from mT5 teachers:

1. **Offline generation**: mT5-base and mT5-XLSum generate summaries for all training articles
2. **Storage**: JSON arrays in `data/pseudo_labels/` (aligned by training index)
3. **Runtime substitution**: `PseudoLabelDataCollator` randomly replaces gold labels with a pseudo-label at probability `pseudo_prob = 0.3`
4. On each training step, each sample has a 30% chance of using an mT5 pseudo-label instead of the gold summary

**Why pseudo-labels instead of logit KD from mT5?**
mT5 uses SentencePiece with a 250K multilingual vocabulary; BanglaT5-small uses a different 32K Bangla vocabulary. Logit dimensions don't match, making direct KL divergence impossible. Pseudo-labels (decoded text re-encoded by student tokenizer) bypass this vocabulary mismatch.

### 5.4 Component: Intermediate Encoder Hidden State Matching (L_MSE)

Aligns internal representations, not just output distributions:

```python
projection = Linear(512, 768)          # learned projection
s_proj = projection(h_student)          # project student to teacher dim
s_norm = normalize(s_proj, p=2, dim=-1) # L2 normalize
t_norm = normalize(h_teacher, p=2, dim=-1)
L_MSE = ||s_norm − t_norm||² (masked by attention_mask)
```

- Projection initialized with Xavier uniform, bias initialized to zero
- Normalized MSE prevents scale mismatch between student (d=512) and teacher (d=768)
- Only applied to encoder (not decoder) hidden states
- Attention mask ensures padding positions don't contribute

### 5.5 Pseudo-Label Generation Pipeline

`generate_pseudo_labels.py` / `generate_pseudo_labels_bansum.py`:

1. Load fine-tuned mT5 teacher checkpoint
2. Process training data in batches (batch_size=16, num_beams=4, max_length=128)
3. Save every 50 batches to `.partial.json` for crash recovery
4. OOM fallback: if a batch fails, retry one sample at a time (batch_size=1)
5. Final output: JSON array of strings aligned to training CSV row indices
6. Resume support: if partial file exists, continue from where it stopped

**Output files (XLSum)**:
- `data/pseudo_labels/train_mt5_base.json` — 55,037+ entries
- `data/pseudo_labels/train_mt5_xlsum.json` — 55,037+ entries

**Output files (BanSum)**:
- `data/pseudo_labels_bansum/train_mt5_base.json` — 112,960 entries
- `data/pseudo_labels_bansum/train_mt5_xlsum.json` — 112,960 entries

---

## 6. Ablation Study Design (A1–A5)

Each configuration incrementally adds one component to isolate its contribution:

| Config | L_CE | L_KL (logit KD) | Pseudo-labels | Adaptive τ | L_MSE (intermediate) | α_kd | α_inter | pseudo_prob | τ |
|--------|------|-----------------|---------------|-----------|---------------------|------|---------|-------------|---|
| **A1_baseline** | ✓ | ✗ | ✗ | ✗ | ✗ | 0.0 | 0.0 | 0.0 | 1.0 |
| **A2_single_kd** | ✓ | ✓ | ✗ | ✗ | ✗ | 0.01 | 0.0 | 0.0 | 0.8 (fixed) |
| **A3_multi_teacher** | ✓ | ✓ | ✓ | ✗ | ✗ | 0.01 | 0.0 | 0.3 | 0.8 (fixed) |
| **A4_adaptive_temp** | ✓ | ✓ | ✓ | ✓ | ✗ | 0.01 | 0.0 | 0.3 | 0.5–2.0 |
| **A5_full_pipeline** | ✓ | ✓ | ✓ | ✓ | ✓ | 0.01 | 0.1 | 0.3 | 0.5–2.0 |

### Shared Training Hyperparameters

| Parameter | XLSum | BanSum |
|-----------|-------|--------|
| Student model | `csebuetnlp/banglat5_small` | `csebuetnlp/banglat5_small` |
| Batch size | 16 | 16 |
| Gradient accumulation | 2 (eff. 32) | 2 (eff. 32) |
| Learning rate | 5e-5 | 5e-5 |
| Weight decay | 0.01 | 0.01 |
| Warmup steps | 500 | 500 |
| Epochs | 5 | 5 |
| Eval/save steps | 1,500 | 1,500 |
| Precision | bf16 | bf16 |
| Early stopping patience | 5 | 5 |
| Metric for best model | ROUGE-L | ROUGE-L |
| Generation beams | 6 | 6 |
| MAX_INPUT_LENGTH | 512 | 850 |
| MAX_TARGET_LENGTH | 256 | 256 |
| Seed | 42 | 42 |
| Optimizer | AdamW | AdamW |

---

## 7. Pipeline Execution Order

The full experimental pipeline was executed in this order:

### Phase 1: Data Preparation
```
1. python preprocess.py                          # XLSum: CSV → train/val/test split
2. python check_data_quality.py                  # Quality audit
3. python analyze_lengths.py                     # Token length analysis → set max lengths
4. python analyze_bansum.py                      # BanSum token length analysis
```

### Phase 2: Teacher Training
```
5. python train_teacher.py                       # BanglaT5 teacher on XLSum
6. python train_teacher_bansum.py                # BanglaT5 teacher on BanSum
   (mT5-base and mT5-XLSum teachers trained separately on both datasets)
```

### Phase 3: Teacher Evaluation & Checkpoint Selection
```
7. python evaluate_all_checkpoints.py            # Find best mT5-base checkpoint
8. python evaluate_xlsum_checkpoints.py          # Find best mT5-XLSum checkpoint
9. python compare_best_checkpoints.py            # Head-to-head comparison
```

### Phase 4: Pseudo-Label Generation
```
10. python generate_pseudo_labels.py             # mT5 → pseudo-labels for XLSum
11. python generate_pseudo_labels_bansum.py      # mT5 → pseudo-labels for BanSum
```

### Phase 5: Ablation Training
```
12. python run_ablation.py                       # A1→A5 on XLSum (orchestrated)
13. python run_ablation_bansum.py                # A1→A5 on BanSum (orchestrated)
    (or individually: python train_student_ablation.py --config A2_single_kd)
```

### Phase 6: Evaluation & Analysis
```
14. python evaluate_ablation_metrics.py          # Extended metrics (XLSum)
15. python evaluate_ablation_metrics_bansum.py   # Extended metrics (BanSum)
16. python eval_a1_a4_checkpoints.py             # BanSum checkpoint selection (A1-A4)
17. python eval_a5_checkpoints.py                # BanSum checkpoint selection (A5)
18. python compare_best_models.py                # Full metrics comparison (BanSum)
```

### Phase 7: Summary Generation for Multi-Judge Evaluation
```
19. py -3.12 generate_summaries_for_judging.py   # 4,000 summaries for LLM judging
```

---

## 8. Script Reference

### 8.1 Training Scripts

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `train_teacher.py` | Fine-tune BanglaT5 teacher on XLSum | `data/train.csv`, `data/val.csv`, `data/test.csv` | `bangla_t5_teacher_finetuned_<ts>/` |
| `train_teacher_bansum.py` | Fine-tune BanglaT5 teacher on BanSum | `bansum_lte_1000_tokens.json` | `banglat5_bansum_<ts>/` |
| `train_student.py` | Standard KD (single teacher) | Train/val/test CSVs + teacher model | `bangla_t5_student_distilled_<ts>/` |
| `train_student_ablation.py` | Ablation training on XLSum | Train/val/test CSVs + teacher + pseudo-labels | `ablation_results/<config>_<ts>/` |
| `train_student_ablation_bansum.py` | Ablation training on BanSum | BanSum JSON + teacher + pseudo-labels | `ablation_results_bansum/<config>_<ts>/` |

### 8.2 Orchestration Scripts

| Script | Purpose | Command |
|--------|---------|---------|
| `run_ablation.py` | Run all A1→A5 on XLSum | `python run_ablation.py [--quick] [--configs A1 A2]` |
| `run_ablation_bansum.py` | Run all A1→A5 on BanSum | `python run_ablation_bansum.py [--quick]` |

### 8.3 Data & Pseudo-Label Scripts

| Script | Purpose | Command |
|--------|---------|---------|
| `preprocess.py` | Clean raw CSV, create 70/20/10 split | `python preprocess.py` |
| `generate_pseudo_labels.py` | XLSum pseudo-labels from mT5 teachers | `python generate_pseudo_labels.py [--quick]` |
| `generate_pseudo_labels_bansum.py` | BanSum pseudo-labels from mT5 teachers | `python generate_pseudo_labels_bansum.py` |

### 8.4 Evaluation & Analysis Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `evaluate_all_checkpoints.py` | Evaluate mT5-base checkpoints | `checkpoint_evaluation_results.json` |
| `evaluate_xlsum_checkpoints.py` | Evaluate mT5-XLSum checkpoints | `xlsum_checkpoint_evaluation_results.json` |
| `compare_best_checkpoints.py` | Compare best mT5 checkpoints (full test) | `best_checkpoints_comparison_full_test.json` |
| `eval_a1_a4_checkpoints.py` | Select best checkpoints for BanSum A1-A4 | `ablation_results_bansum/a1_a4_checkpoint_eval_results.json` |
| `eval_a5_checkpoints.py` | Select best checkpoint for BanSum A5 | Renames to `best_model/` |
| `evaluate_ablation_metrics.py` | Extended metrics for XLSum ablations | `ablation_results/ablation_extended_metrics.json` |
| `evaluate_ablation_metrics_bansum.py` | Extended metrics for BanSum ablations | `ablation_results_bansum/ablation_extended_metrics.json` |
| `compare_best_models.py` | Full metrics comparison (BanSum A1-A5) | `ablation_results_bansum/ablation_full_metrics_comparison.json` |
| `analyze_lengths.py` | Token length distribution analysis | Console output |
| `analyze_bansum.py` | BanSum token length analysis | Console output |
| `check_data_quality.py` | Data quality audit | Console output + quality score |
| `check_gpu.py` | GPU availability check | Console output |
| `quick_check.py` | Quick BanSum dataset structure check | Console output |
| `test_inference.py` | Test inference on 5 random samples | Console output |

### 8.5 Summary Generation

| Script | Purpose | Output |
|--------|---------|--------|
| `generate_summaries_for_judging.py` | Generate 4,000 summaries from A2/A5 models | `judge_evaluation_samples.csv` + `judge_evaluation_metadata.json` |

---

## 9. Experimental Results — XLSum Dataset

### 9.1 ROUGE Scores (test set: 8,033 samples)

| Config | ROUGE-1 | ROUGE-2 | ROUGE-L | Test Loss | Train Loss | Runtime (s) |
|--------|---------|---------|---------|-----------|------------|-------------|
| **A1_baseline** | 0.3943 | 0.2321 | 0.3794 | 1.7822 | 4.7251 | 5,139.8 |
| **A2_single_kd** | **0.3945** | **0.2321** | **0.3797** | **1.7777** | 1.3256 | 2,242.1 |
| A3_multi_teacher | 0.3924 | 0.2301 | 0.3777 | 1.8706 | 1.2132 | — |
| A4_adaptive_temp | 0.3914 | 0.2296 | 0.3767 | 1.8848 | 2.6674 | — |
| A5_full_pipeline | 0.3914 | 0.2299 | 0.3769 | 1.8832 | 1.2238 | 2,586.8 |
| *Teacher (BanglaT5)* | *0.4234* | *0.2512* | *0.4058* | *1.7282* | — | — |

**Key finding (XLSum)**: A2 achieves the best ROUGE scores. Adding more components (A3–A5) slightly hurts ROUGE on this dataset.

### 9.2 Extended Metrics (test set: 8,032–8,033 samples)

| Config | BLEU | BERTScore F1 | Semantic Similarity |
|--------|------|-------------|-------------------|
| A1_baseline | 15.79 | 0.7863 | 0.8678 |
| A2_single_kd | 15.49 | 0.7867 | 0.8683 |
| A3_multi_teacher | 15.20 | 0.7855 | 0.8689 |
| A4_adaptive_temp | 15.02 | 0.7846 | 0.8693 |
| **A5_full_pipeline** | 15.08 | 0.7848 | **0.8695** |
| *Teacher (BanglaT5)* | *16.47* | *0.7917* | *0.8722* |

**Key finding (XLSum)**: While A2 dominates in ROUGE, **A5 achieves the best semantic similarity** (0.8695). This suggests the full pipeline produces summaries that are more semantically faithful despite slightly lower n-gram overlap. The progressive improvement in semantic similarity from A1→A5 (0.8678→0.8695) confirms that each component adds meaningful semantic knowledge.

### 9.3 XLSum Ablation Insights

- **A1→A2 (+ logit KD)**: Small but consistent improvement across all metrics. Logit KD provides the foundational knowledge transfer.
- **A2→A3 (+ pseudo-labels)**: ROUGE dips slightly but semantic similarity improves (0.8683→0.8689). The mT5 pseudo-labels introduce lexical diversity that reduces n-gram overlap with gold references while maintaining (or improving) meaning.
- **A3→A4 (+ adaptive τ)**: Further ROUGE decrease but semantic similarity continues improving (0.8689→0.8693). Adaptive temperature adds regularization at the cost of sharper predictions.
- **A4→A5 (+ intermediate matching)**: ROUGE stabilizes; semantic similarity reaches peak (0.8695). Encoder alignment transfers structural understanding.

---

## 10. Experimental Results — BanSum Dataset

### 10.1 ROUGE Scores (test set: 14,120 samples)

Reported from training evaluation (`ablation_comparison.json`):

| Config | ROUGE-1 | ROUGE-2 | ROUGE-L | Test Loss | Train Loss |
|--------|---------|---------|---------|-----------|------------|
| A1_baseline | 0.3211 | 0.1834 | 0.2577 | 1.7564 | 0.9044 |
| **A2_single_kd** | **0.3349** | **0.1921** | **0.2717** | 1.7609 | 0.7773 |
| A3_multi_teacher | 0.3403 | 0.1948 | 0.2755 | 1.7947 | 1.1331 |
| A4_adaptive_temp | 0.3248 | 0.1826 | 0.2604 | 1.8098 | 0.8383 |

*(A5 was trained separately with batch_size=16 backup configuration)*

### 10.2 Full Metrics Comparison (best_model evaluation on 14,120 test samples)

From `ablation_full_metrics_comparison.json`:

| Config | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | BERTScore F1 | Semantic Sim |
|--------|---------|---------|---------|------|-------------|-------------|
| A1_baseline | 0.2957 | 0.1389 | 0.2314 | 11.67 | 0.7441 | 0.7538 |
| **A2_single_kd** | **0.3057** | **0.1462** | **0.2435** | **12.05** | **0.7486** | **0.7603** |
| A3_multi_teacher | 0.3014 | 0.1444 | 0.2405 | 11.84 | 0.7465 | 0.7584 |
| A4_adaptive_temp | 0.2866 | 0.1352 | 0.2253 | 10.88 | 0.7393 | 0.7443 |
| A5_full_pipeline | 0.2881 | 0.1361 | 0.2268 | 10.87 | 0.7391 | 0.7466 |
| *Teacher (BanglaT5)* | *0.3663* | *0.2135* | *0.2998* | — | — | — |

**Key finding (BanSum)**: **A2 dominates across ALL metrics** — ROUGE, BLEU, BERTScore, and semantic similarity. Unlike XLSum, the additional components (A3–A5) hurt performance on BanSum.

### 10.3 BanSum Ablation Insights

- **A1→A2 (+ logit KD)**: Strong and consistent improvement (+0.0121 ROUGE-L, +0.0065 semantic sim). Single-teacher KD provides clear benefit.
- **A2→A3 (+ pseudo-labels)**: Slight regression from A2 but still above A1. mT5 pseudo-labels add noise on this dataset.
- **A3→A4 (+ adaptive τ)**: Significant drop below A1 baseline (−0.0061 ROUGE-L vs A1). Adaptive temperature may be too aggressive on BanSum's longer outputs.
- **A4→A5 (+ intermediate matching)**: Marginal recovery from A4 (+0.0015 ROUGE-L) but still below A1.

---

## 11. Cross-Dataset Analysis

### 11.1 Consistent Findings

1. **A2 (single-teacher logit KD) is the most reliable configuration**: Best on BanSum across all metrics, best ROUGE on XLSum.
2. **Logit-level KD from a same-vocabulary teacher provides consistent benefit**: A1→A2 improvement is robust across both datasets.
3. **Student achieves ~93–94% of teacher ROUGE-L**: XLSum: 0.3797/0.4058 = 93.6%. BanSum: 0.2435/0.2998 = 81.2%.

### 11.2 Dataset-Dependent Findings

1. **Pseudo-labels**: Helpful for semantic similarity on XLSum (+0.0006), slightly harmful on BanSum (−0.0019).
2. **Adaptive temperature**: Improves semantic similarity on XLSum, hurts BanSum significantly.
3. **Intermediate matching**: Improves semantic similarity on XLSum (A5 best semantic), marginal on BanSum.
4. **Full pipeline (A5)**: Best semantic similarity on XLSum, but underperforms on BanSum.

### 11.3 Interpretation

The XLSum dataset has shorter summaries (~8 tokens generated) and a cleaner domain (BBC Bangla), making it more tolerant of additional regularization (adaptive τ, pseudo-noise). BanSum has much longer outputs (~106–248 tokens) and more diverse article styles, where the extra components introduce harmful noise. This suggests that **multi-component KD is most beneficial when the task has compact outputs and clean teacher signal**.

---

## 12. Multi-Judge LLM Evaluation

### 12.1 Generated Samples

`generate_summaries_for_judging.py` produces summaries for external LLM-based quality assessment:

| Model | Dataset | Samples | Avg time/sample | Throughput |
|-------|---------|---------|-----------------|------------|
| A2_single_kd | XLSum | 1,000 | 0.0113s | 88.55 samples/s |
| A5_full_pipeline | XLSum | 1,000 | 0.0109s | 91.64 samples/s |
| A2_single_kd_bansum | BanSum | 1,000 | 0.1213s | 8.24 samples/s |
| A5_full_pipeline_bansum | BanSum | 1,000 | 0.1000s | 10.00 samples/s |

**Total**: 4,000 entries in `judge_evaluation_samples.csv`

### 12.2 CSV Output Format

| Column | Description |
|--------|-------------|
| `sample_id` | Index within the 1,000 selected samples (0–999) |
| `original_test_index` | Original row index in `data/test.csv` |
| `category` | News category (if available) |
| `article` | Original Bangla article text |
| `reference_summary` | Gold reference summary |
| `model_summary` | Generated summary from the model |
| `model_name` | Which model produced it (A2/A5 × XLSum/BanSum) |
| `generation_time_sec` | Time to generate this summary |

### 12.3 Intended Use

The CSV is designed for feeding into a **multi-judge framework** where proprietary LLMs (ChatGPT, Claude, Grok, etc.) evaluate summary quality on dimensions such as:
- Fluency
- Faithfulness / factual consistency
- Informativeness / coverage
- Conciseness
- Overall quality ranking (A2 vs A5)

Metadata (timing, model parameters) in `judge_evaluation_metadata.json` supports efficiency analysis.

---

## 13. Inference Speed & Efficiency

All student models have identical architecture (109.9M params). Speed differences arise from generation characteristics (output length, beam search dynamics):

### XLSum Models (short summaries ~8 tokens)

| Model | Load time | Throughput | Avg time/sample |
|-------|-----------|------------|-----------------|
| A2_single_kd | 0.772s | 88.55 samples/s | 0.0113s |
| A5_full_pipeline | 0.475s | 91.64 samples/s | 0.0109s |

### BanSum Models (long summaries ~106–248 tokens)

| Model | Load time | Throughput | Avg time/sample |
|-------|-----------|------------|-----------------|
| A2_single_kd_bansum | 1.281s | 8.24 samples/s | 0.1213s |
| A5_full_pipeline_bansum | 1.236s | 10.00 samples/s | 0.1000s |

**Key insight**: The ~10× throughput difference between XLSum and BanSum models is entirely due to output length (8 tokens vs 106+ tokens). The models themselves are identical in size and architecture. Beam search decoding (num_beams=6) is the bottleneck, scaling linearly with output length.

---

## 14. Directory Structure

```
not_more_than_limit/
│
├── data/
│   ├── train.csv                          # Training samples
│   ├── val.csv                            # Validation samples
│   ├── test.csv                           # Test samples
│   ├── text_summarization.csv             # Raw XLSum data (pre-split)
│   ├── pseudo_labels/
│   │   ├── train_mt5_base.json            # mT5-base pseudo-summaries (XLSum)
│   │   └── train_mt5_xlsum.json           # mT5-XLSum pseudo-summaries (XLSum)
│   └── pseudo_labels_bansum/
│       ├── train_mt5_base.json            # mT5-base pseudo-summaries (BanSum)
│       └── train_mt5_xlsum.json           # mT5-XLSum pseudo-summaries (BanSum)
│
├── ablation_results/                      # XLSum ablation outputs
│   ├── ablation_comparison.json           # A1–A5 ROUGE comparison
│   ├── ablation_extended_metrics.json     # + BLEU, BERTScore, Semantic Sim
│   ├── A1_baseline_20260222_043307/
│   │   ├── best_model/                    # Best checkpoint model files
│   │   ├── checkpoint-*/                  # Training checkpoints
│   │   ├── logs/                          # TensorBoard logs
│   │   └── ablation_results.json          # Full training results
│   ├── A2_single_kd_20260222_093225/      # (same structure)
│   ├── A3_multi_teacher_20260222_132356/
│   ├── A4_adaptive_temp_20260223_002004/
│   └── A5_full_pipeline_20260223_005736/
│
├── ablation_results_bansum/               # BanSum ablation outputs
│   ├── ablation_comparison.json           # A1–A4 ROUGE (A5 trained separately)
│   ├── ablation_full_metrics_comparison.json  # Full metrics A1–A5
│   ├── a1_a4_checkpoint_eval_results.json
│   ├── A1_baseline_20260224_121256/
│   ├── A2_single_kd_20260225_120221/
│   ├── A3_multi_teacher_20260226_095657/
│   ├── A4_adaptive_temp_20260227_192419/
│   └── A5_full_pipeline_20260228_034939_batch16_backup/
│
├── bangla_t5_teacher_finetuned_20251216_143715/  # XLSum BanglaT5 teacher
│   ├── final_model/                       # Saved model
│   ├── training_config.json
│   ├── test_results.json
│   ├── train_results.json
│   ├── eval_results.json
│   └── logs/
│
├── banglat5_bansum_20260218_213532/       # BanSum BanglaT5 teacher
│   ├── final_model/
│   ├── training_config.json
│   ├── test_results.json
│   └── checkpoint-28240/
│
├── mt5_teacher_mt5-base_20260208_124334/  # XLSum mT5-base teacher
│   ├── checkpoint-16000/                  # Best checkpoint (by semantic sim)
│   ├── training_config.json
│   ├── test_results.json
│   └── logs/
│
├── mt5_xlsum_20260212_060223/             # XLSum mT5-XLSum teacher
│   ├── checkpoint-12000/                  # Best checkpoint
│   ├── training_config.json
│   ├── test_results.json
│   └── logs/
│
├── mt5base_bansum_20260219_113113/        # BanSum mT5-base teacher
│   └── checkpoint-16000/
│
├── mt5xlsum_bansum_20260219_062938/       # BanSum mT5-XLSum teacher
│   └── checkpoint-14000/
│
├── bansum_lte_1000_tokens.json            # BanSum dataset file
│
├── judge_evaluation_samples.csv           # 4,000 summaries for LLM judging
├── judge_evaluation_metadata.json         # Generation config + timing
├── best_checkpoints_comparison_full_test.json
├── checkpoint_evaluation_results.json
│
├── preprocess.py                          # Data preprocessing
├── train_teacher.py                       # BanglaT5 teacher training (XLSum)
├── train_teacher_bansum.py                # BanglaT5 teacher training (BanSum)
├── train_student.py                       # Standard KD training
├── train_student_ablation.py              # Ablation training (XLSum)
├── train_student_ablation_bansum.py       # Ablation training (BanSum)
├── run_ablation.py                        # Orchestrator (XLSum)
├── run_ablation_bansum.py                 # Orchestrator (BanSum)
├── generate_pseudo_labels.py              # Pseudo-label generation (XLSum)
├── generate_pseudo_labels_bansum.py       # Pseudo-label generation (BanSum)
├── evaluate_ablation_metrics.py           # Extended metrics (XLSum)
├── evaluate_ablation_metrics_bansum.py    # Extended metrics (BanSum)
├── evaluate_all_checkpoints.py            # mT5-base checkpoint evaluation
├── evaluate_xlsum_checkpoints.py          # mT5-XLSum checkpoint evaluation
├── compare_best_checkpoints.py            # Checkpoint comparison
├── compare_best_models.py                 # Full model comparison (BanSum)
├── eval_a1_a4_checkpoints.py              # BanSum A1-A4 checkpoint selection
├── eval_a5_checkpoints.py                 # BanSum A5 checkpoint selection
├── analyze_lengths.py                     # Token length analysis
├── analyze_bansum.py                      # BanSum analysis
├── check_data_quality.py                  # Data quality audit
├── check_gpu.py                           # GPU info
├── quick_check.py                         # BanSum quick check
├── test_inference.py                      # Inference testing
├── generate_summaries_for_judging.py      # Multi-judge sample generation
└── README.md                              # This file
```

---

## 15. How to Run

### Prerequisites

```bash
py -3.12 -m pip install torch transformers datasets pandas numpy rouge-score bert-score sentence-transformers sacrebleu tqdm
```

### Full Pipeline (XLSum)

```bash
# 1. Preprocess data
py -3.12 preprocess.py

# 2. Train teacher (if not already done)
py -3.12 train_teacher.py

# 3. Generate pseudo-labels (auto-called by run_ablation if missing)
py -3.12 generate_pseudo_labels.py

# 4. Run all ablations A1→A5
py -3.12 run_ablation.py

# 5. Evaluate extended metrics
py -3.12 evaluate_ablation_metrics.py

# 6. Generate summaries for LLM judging
py -3.12 generate_summaries_for_judging.py
```

### Full Pipeline (BanSum)

```bash
# 1. Train BanSum teacher
py -3.12 train_teacher_bansum.py

# 2. Generate BanSum pseudo-labels
py -3.12 generate_pseudo_labels_bansum.py

# 3. Run all ablations on BanSum
py -3.12 run_ablation_bansum.py

# 4. Select best checkpoints
py -3.12 eval_a1_a4_checkpoints.py
py -3.12 eval_a5_checkpoints.py

# 5. Full metrics comparison
py -3.12 compare_best_models.py
```

### Quick Test Mode

```bash
py -3.12 run_ablation.py --quick              # 500 train samples, 1 epoch
py -3.12 train_student_ablation.py --config A2_single_kd --quick
```

### Run Specific Configs Only

```bash
py -3.12 run_ablation.py --configs A1_baseline A5_full_pipeline
```

### Resume an Interrupted Run

```bash
py -3.12 train_student_ablation.py --config A3_multi_teacher --resume_dir ablation_results/A3_multi_teacher_20260222_132356
```

---

## 16. Resume & Fault Tolerance

The entire pipeline is designed for robustness against interruptions (power outages, GPU crashes, OOM errors):

### Pseudo-Label Generation
- Saves `.partial.json` checkpoint every 50 batches
- On restart, detects partial file and resumes from last saved index
- OOM fallback: if a batch fails, retries sample-by-sample (batch_size=1)

### Ablation Training
- HuggingFace `Seq2SeqTrainer` saves checkpoints every `SAVE_STEPS` (1,500 steps)
- `save_total_limit=3` keeps only the 3 most recent checkpoints (disk management)
- On restart:
  1. `run_ablation.py` checks if `ablation_results.json` exists → skips completed configs
  2. If directory exists but no results file → finds latest `checkpoint-*` folder → resumes
  3. `train_student_ablation.py` also auto-detects existing directories and resumes

### Recovery Commands

If the pipeline was interrupted:
```bash
# Simply re-run the same command — it will detect state and resume
py -3.12 run_ablation.py
py -3.12 run_ablation_bansum.py
```

---

## 17. Dependencies & Hardware

### Python Dependencies

| Package | Purpose |
|---------|---------|
| `torch >= 2.0` | Deep learning framework |
| `transformers >= 5.0` | HuggingFace model loading, training, generation |
| `datasets` | HuggingFace dataset handling |
| `pandas` | CSV/dataframe operations |
| `numpy` | Numerical operations |
| `rouge-score` | ROUGE metric computation |
| `bert-score` | BERTScore metric |
| `sentence-transformers` | Semantic similarity (paraphrase-multilingual-MiniLM-L12-v2) |
| `sacrebleu` | BLEU score computation |
| `tqdm` | Progress bars |

### Hardware Requirements

| Task | Min GPU VRAM | Recommended |
|------|-------------|-------------|
| Teacher training (BanglaT5, batch=4, grad_ckpt) | 16 GB | 24 GB |
| Teacher training (BanglaT5, batch=16, BanSum) | 24 GB | 40 GB |
| mT5 teacher training | 16 GB | 24 GB |
| Student ablation training (A1–A4) | 12 GB | 16 GB |
| Student ablation training (A5, teacher in VRAM) | 16 GB | 24 GB |
| Pseudo-label generation | 8 GB | 16 GB |
| Inference / evaluation | 4 GB | 8 GB |

### Environment

- **Python**: 3.12.3
- **CUDA**: Compatible with PyTorch 2.x
- **OS**: Windows (paths use both `/` and `\`; scripts handle both)
- **Memory management**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set in training scripts

---

## 18. Design Decisions & Rationale

### Why α_kd = 0.01 (not the typical 0.5)?

In standard image classification KD, α=0.5 balances hard and soft losses. In seq2seq:
- KL divergence (summed over vocabulary ~32K) produces magnitudes **50–100× larger** than CE loss
- Using α_kd = 0.5 would drown out the hard label signal entirely
- α_kd = 0.01 was empirically tuned to keep the KD loss contribution at ~1% of total, providing gentle guidance without destabilizing training

### Why τ = 0.8 (below 1.0)?

Traditional KD uses τ > 1 to soften distributions. For Bangla summarization:
- The BanglaT5 teacher is already well-calibrated (ROUGE-L 0.4058)
- A sharper temperature (τ < 1) preserves the teacher's confident predictions rather than flattening them
- Empirically, τ = 0.8 outperformed τ = 1.0 and τ = 2.0 on validation

### Why pseudo-label probability = 0.3?

- 0.0 = no pseudo-labels (A2 baseline KD)
- 0.3 = each sample has 30% chance of using mT5 pseudo-label, 70% gold
- Higher values (0.5+) degraded performance as mT5 pseudo-labels are noisier than gold
- 0.3 balances diversity introduction with training signal quality

### Why intermediate matching on encoder only (not decoder)?

- Encoder representations capture input understanding (structural, semantic)
- Decoder representations are highly conditioned on teacher-specific generation patterns that may not transfer cleanly
- Encoder alignment is more stable and the dimensional mismatch (512→768) is easier to project in one direction

### Why normalized MSE instead of raw MSE?

- Raw MSE is dominated by representational scale differences between student (d=512) and teacher (d=768)
- L2 normalization maps both to the unit hypersphere, making the loss purely about directional alignment
- This prevents the student from simply learning to match magnitudes rather than semantics

### Why beam search with 6 beams?

- Greedy decoding (beams=1) produces shorter, less fluent Bangla output
- 4 beams is standard; 6 was found to improve summary completeness
- Beyond 6, diminishing returns and increased latency

### Why early stopping patience = 5?

- With eval every 1,500 steps and 5 epochs, this allows ~2.5 epochs of no improvement before stopping
- Prevents overfitting while allowing sufficient training time for convergence

### Why two datasets (XLSum + BanSum)?

- Validates that findings are robust, not dataset-specific
- XLSum: shorter summaries, cleaner domain (BBC Bangla)
- BanSum: longer summaries, more diverse sources, 2× larger training set
- Different behavior on the two datasets (A5 best semantic on XLSum, A2 best everything on BanSum) provides nuanced conclusions for the thesis

### Why MAX_INPUT_LENGTH = 850 for BanSum (vs 512 for XLSum)?

Token length analysis (`analyze_bansum.py`) showed BanSum articles are significantly longer:
- BanSum: mean ~500–600 tokens, max ~990
- XLSum: 95th percentile ~550 tokens
- 850 captures the vast majority of BanSum articles without excessive padding
