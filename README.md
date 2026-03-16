# MultiTeacher-TokenGating-Summarization

Multi-teacher knowledge distillation framework for low-resource abstractive summarization. This repository includes token-gating strategies, agreement-aware distillation, and a multilingual offline pseudo-labeling pipeline that compresses large teachers into smaller students while retaining most of the quality.

## Highlights

- Multi-teacher distillation with agreement-aware token distillation (EWAD) and capacity-proportional divergence preservation (CPDP).
- Token-gating and multi-teacher routing scripts for low-resource summarization.
- Multilingual teacher-to-student compression pipeline using offline pseudo-labeling.
- Evaluation utilities for ROUGE, BLEU, BERTScore, and semantic similarity.

## What Is In This Repo

- Training and evaluation scripts for multi-teacher token-gating summarization.
- Multilingual distillation scripts (teacher fine-tune, pseudo-labeling, student training).
- Result JSONs that summarize evaluation metrics.
- No datasets or model checkpoints are stored here.

## Multilingual Distillation Pipeline (Offline)

The multilingual pipeline uses offline pseudo-labeling to avoid running the teacher on every step.

1) Preprocess and split datasets

```bash
python preprocess_multilingual.py
python count_and_create_datasets.py
```

2) Fine-tune a teacher per language

```bash
python setup_language.py hindi
python train_teacher.py
```

3) Generate pseudo-labels (teacher summaries cached once)

```bash
python generate_teacher_labels.py
```

4) Train the student on cached labels

```bash
python train_student_fast.py
```

5) Automated multi-language pipeline

```bash
python train_remaining_languages.py
```

## Results

See the summary metrics in training_results.json. Additional evaluation artifacts may be present in checkpoint_evaluation_results.json or pipeline_eval_results.json.

## Setup

```bash
pip install -r requirements.txt
```

## Notes

- Datasets are not included. Download and preprocess them locally (XLSum or other supported corpora).
- Model checkpoints are not stored in this repo.
- If you are only using the multilingual pipeline, the core scripts are the teacher training, label generation, and student training files listed above.

## Citation

If you use this repo in academic work, please cite the project or link to this repository.
