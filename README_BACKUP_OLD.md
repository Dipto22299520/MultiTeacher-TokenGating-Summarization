# BanglaT5 MapReduce Summarization System

A sophisticated two-stage MapReduce summarization pipeline for long Bangla news articles using fine-tuned BanglaT5 (296M parameter) transformer models.

## 🎯 Problem Statement

Standard transformer models have a context window limitation (~1024 tokens for T5). Many news articles exceed this limit, requiring a different approach. This project implements a **MapReduce** architecture that:
1. Splits long articles into manageable chunks
2. Summarizes each chunk independently (MAP)
3. Merges chunk summaries into a coherent final summary (REDUCE)

## 🏗️ Architecture

### Two-Stage Pipeline

```
Long Article (>1024 tokens)
    ↓
[CHUNK] → Sentence-aligned chunks (~900 tokens with overlap)
    ↓
[MAP] → Summarize each chunk (Model 1: Salience Ranking)
    ↓
[REDUCE] → Merge chunk summaries (Model 2: Coherence & Deduplication)
    ↓
Final Summary
```

### Model Components

| Component | Model Path | Purpose | Training Data |
|-----------|-----------|---------|---------------|
| **MAP Model** | `banglaT5_full_doc_20260215_123349/checkpoint-7000` | Extract salient facts from each chunk | Full documents (first 1024 tokens → gold summary) |
| **REDUCE Model** | `banglaT5_reduce_task_20260217_111025/checkpoint-6000` | Merge and deduplicate chunk summaries | Multiple summaries → final summary |

## 📊 Dataset

- **Source**: XL-Sum Bangla news dataset
- **Training Split**: Articles with >1000 tokens (1,600 samples after cleanup)
- **Test Split**: 1,988 articles
- **Domain**: Bangla news articles from BBC Bangla

### Data Processing

```python
# Articles split by length
├── bangla_train_lte_1000.json    # ≤1000 tokens (short articles)
├── bangla_train_gt_1000.json     # >1000 tokens (long articles)
└── data_splits/
    ├── train.json                 # Training set
    ├── val.json                   # Validation set
    └── test.json                  # Test set (1,988 articles)
```

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate    # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- `transformers` - Hugging Face transformers for T5 models
- `torch` - PyTorch for model inference
- `sentence-transformers` - Semantic similarity evaluation
- `bert-score` - BERTScore metric
- `rouge-score` - ROUGE metrics
- `nltk` - BLEU score computation

### Usage

#### Run Pipeline on Sample Articles

```bash
python run_pipeline.py
```

#### Evaluate on Full Test Set

```bash
python eval_pipeline.py
```

#### Summarize Custom Text

```python
from run_pipeline import ChunkedSummarizer

summarizer = ChunkedSummarizer(
    map_model_path="./banglaT5_full_doc_20260215_123349/checkpoint-7000",
    reduce_model_path="./banglaT5_reduce_task_20260217_111025/checkpoint-6000"
)

result = summarizer.summarize("আপনার বাংলা নিবন্ধ এখানে...")
print(result['summary'])
print(f"Method used: {result['method']}")  # single_pass / single_chunk / map_reduce
print(f"Number of chunks: {result['num_chunks']}")
```

## 🔧 Training Process

### Stage 1: MAP Model Training

**Objective**: Teach the model to identify salient information

```bash
python train_full_document.py
```

- **Dataset**: Full Bangla news articles
- **Input**: First 1024 tokens of article + prefix "summarize bangla news: "
- **Output**: Gold standard summary
- **Learning**: Model learns which facts are important (salience ranking)
- **Best Checkpoint**: checkpoint-7000
  - ROUGE-L: 0.188
  - BERTScore F1: 0.731
  - Semantic Similarity: 0.754

### Stage 2: REDUCE Model Training

**Objective**: Merge multiple summaries coherently

```bash
python train_reduce_task.py
```

- **Dataset**: Generated from chunk-level summaries
- **Input**: Concatenated chunk summaries + prefix "summarize multiple summaries: "
- **Output**: Final merged summary
- **Learning**: Deduplication, coherence, information synthesis
- **Best Checkpoint**: checkpoint-6000 (selected after 15 epochs)

## 📈 Evaluation Results

### MAP Model Performance (checkpoint-7000)

| Metric | Score |
|--------|-------|
| ROUGE-1 | 0.329 |
| ROUGE-2 | 0.136 |
| ROUGE-L | **0.188** |
| BERTScore F1 | **0.731** |
| Semantic Similarity | **0.754** |
| BLEU | 15.98 |

### Pipeline Strategy Distribution

On the test set (1,988 articles):

```
Single Pass:   ~1% (Articles < 900 tokens - direct summarization)
Single Chunk:  ~5% (Articles fit in one chunk)
MapReduce:     ~94% (Articles requiring multi-chunk processing)
```

## 🔍 Key Features

### 1. Intelligent Chunking

- **Sentence-aligned**: Never breaks sentences mid-way
- **Overlap**: 3 sentences overlap between chunks for context
- **Size**: ~900 tokens per chunk (safely below 1024 limit)

### 2. Three-Mode Strategy

The system automatically selects the best approach:

```python
if tokens < 900:
    # SINGLE_PASS: Direct summarization
    summary = model.generate(article)
    
elif tokens < 1024:
    # SINGLE_CHUNK: One chunk, no merge needed
    summary = model.generate(chunk)
    
else:
    # MAP_REDUCE: Multi-chunk processing
    chunk_summaries = [model.generate(chunk) for chunk in chunks]
    summary = model.generate(concatenate(chunk_summaries))
```

### 3. BF16 Inference

- Uses `torch.bfloat16` for reduced memory usage
- Enables efficient processing on consumer GPUs

## 📁 Project Structure

```
.
├── run_pipeline.py              # Main pipeline implementation
├── eval_pipeline.py             # Full test-set evaluation
├── train_full_document.py       # Train MAP model
├── train_reduce_task.py         # Train REDUCE model
├── generate_reduce_data.py      # Create REDUCE training data
├── bangla_sentence_splitter.py  # Sentence tokenization
├── chunk_processor.py           # Chunking logic
├── requirements.txt             # Python dependencies
│
├── data_splits/                 # Train/val/test splits
│   ├── train.json
│   ├── val.json
│   └── test.json
│
├── reduce_data/                 # REDUCE model training data
│   ├── reduce_train.json
│   ├── reduce_val.json
│   └── reduce_test.json
│
├── banglaT5_full_doc_*/         # MAP model checkpoints
│   └── checkpoint-7000/         # Best MAP model
│
└── banglaT5_reduce_task_*/      # REDUCE model checkpoints
    └── checkpoint-6000/         # Best REDUCE model
```

## 🎓 Training Configurations

### MAP Model

```python
{
    "model": "csebuetnlp/banglat5",
    "max_source_length": 1024,
    "max_target_length": 256,
    "learning_rate": 3e-4,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_epochs": 10,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "fp16": True
}
```

### REDUCE Model

```python
{
    "model": "csebuetnlp/banglat5",
    "max_source_length": 1024,
    "max_target_length": 256,
    "learning_rate": 3e-4,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_epochs": 15,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "fp16": True
}
```

## 🧪 Evaluation Scripts

### Compare Checkpoints

```bash
# Evaluate all MAP model checkpoints
python eval_checkpoint.py

# Evaluate REDUCE model checkpoints
python eval_reduce_checkpoints.py
```

### Sample Predictions

```bash
# Generate and inspect sample outputs
python sample_predictions.py
```

### Quick Evaluation

```bash
# Fast evaluation on subset
python quick_eval.py
```

## 🐛 Known Issues & Solutions

### Issue 1: Corrupted Training Samples

**Problem**: Some samples had repetitive text with `[CHUNK]` markers (695 repetitions)

**Detection**: Use `verify_data.py` to find corrupted samples

**Solution**: Clean dataset before training

### Issue 2: Progress Bar Not Updating

**Problem**: `tqdm` progress bars don't update in background terminal

**Solution**: Check CPU usage and checkpoint files to verify progress

### Issue 3: Memory Issues

**Problem**: Large models on consumer GPUs

**Solution**: Use BF16 inference (`torch.bfloat16`)

## 📊 Metrics Explained

| Metric | What It Measures | Good Score |
|--------|------------------|------------|
| **ROUGE-L** | Longest common subsequence overlap | > 0.15 |
| **BERTScore** | Semantic similarity using BERT embeddings | > 0.70 |
| **Semantic Similarity** | Cosine similarity of sentence embeddings | > 0.70 |
| **BLEU** | N-gram precision (less reliable for abstractive) | 10-20 |

**Note**: For abstractive summarization, ROUGE scores are typically lower than extractive methods. BERTScore and Semantic Similarity are better indicators of quality.

## 🎯 Research Contributions

1. **MapReduce for Bangla**: First implementation of MapReduce summarization for Bangla language
2. **Dual-Model Architecture**: Specialized models for salience extraction and merging
3. **Sentence-Aligned Chunking**: Maintains linguistic coherence during chunking
4. **Comprehensive Evaluation**: Multiple metrics (ROUGE, BERTScore, Semantic Similarity)

## 📝 Citation

If you use this work, please cite:

```bibtex
@misc{banglat5-mapreduce-2026,
  title={BanglaT5 MapReduce Summarization System for Long News Articles},
  author={Your Name},
  year={2026},
  note={Two-stage MapReduce architecture for Bangla text summarization}
}
```

## 🔗 References

- **BanglaT5**: Bhattacharjee et al. (2022) - [csebuetnlp/banglat5](https://huggingface.co/csebuetnlp/banglat5)
- **XL-Sum Dataset**: Hasan et al. (2021) - Cross-lingual summarization dataset
- **T5 Architecture**: Raffel et al. (2020) - Text-to-Text Transfer Transformer

## 🤝 Contributing

This is a research project. For questions or collaboration:
1. Review the code in `run_pipeline.py` for pipeline logic
2. Check `train_full_document.py` for training procedures
3. See `eval_pipeline.py` for evaluation methodology

## 📄 License

This project uses the BanglaT5 model which is under the MIT License. Check model card for details.

---

**Last Updated**: February 2026  
**Status**: Research Project - Completed Training & Evaluation

