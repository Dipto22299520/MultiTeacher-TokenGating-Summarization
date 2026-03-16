"""
Configuration for Dual-Teacher Knowledge Distillation Pipeline
==============================================================
Qwen2.5-32B (Teacher 1) + Qwen2.5-14B (Teacher 2) → Qwen2.5-3B (Student)
Bengali Abstractive Summarization on BanSum dataset
"""

import os

# ============================================================================
# TEST MODE — Set True to run a quick sanity check with 1000 samples
# ============================================================================
TEST_MODE = False  # <-- Flip to True for quick pipeline test, False for full run

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)  # Parent "not more than limit" dir

DATASET_FILE = os.path.join(BASE_DIR, "bansum_lte_1000_tokens.json")
TEACHER_32B_OUTPUTS = os.path.join(BASE_DIR, "teacher_outputs", "teacher_32b")
TEACHER_14B_OUTPUTS = os.path.join(BASE_DIR, "teacher_outputs", "teacher_14b")
STUDENT_OUTPUT_DIR = os.path.join(BASE_DIR, "student_outputs")
EVAL_OUTPUT_DIR = os.path.join(BASE_DIR, "eval_results")

# ============================================================================
# MODEL NAMES (Qwen2.5 Family — same tokenizer, same architecture)
# ============================================================================
TEACHER_32B_MODEL = "Qwen/Qwen2.5-32B-Instruct"
TEACHER_14B_MODEL = "Qwen/Qwen2.5-14B-Instruct"
STUDENT_MODEL = r"D:\hf_cache\models--Qwen--Qwen2.5-3B\snapshots\3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"

# Quantization for teachers (4-bit to fit on RTX 5090 32GB)
TEACHER_QUANTIZATION = "4bit"  # Options: "4bit", "8bit", "none"
TEACHER_LOAD_DTYPE = "bfloat16"

# ============================================================================
# DATASET
# ============================================================================
DATASET_TEXT_KEY = "main"
DATASET_SUMMARY_KEY = "sum2"
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
SEED = 42

# Limit samples for faster teacher generation (set None for full dataset)
# For testing pipeline: 100, for real run: None
MAX_SAMPLES = None

# ============================================================================
# TEST MODE OVERRIDES — applied at bottom of file after all defaults are set
# ============================================================================
# (see bottom of file)

# ============================================================================
# TEACHER GENERATION CONFIG
# ============================================================================
TEACHER_MAX_INPUT_TOKENS = 1024
TEACHER_MAX_OUTPUT_TOKENS = 256
TEACHER_TEMPERATURE = 0.7  # For summary generation
TEACHER_TOP_P = 0.9
TEACHER_BATCH_SIZE = 2  # Batch size for teacher inference (RTX 5080 16GB)

# For logit extraction (token-level probabilities)
LOGIT_TOP_K = 50  # Save top-k logits per token (saves disk space)

# Prompt template for teacher summary generation
TEACHER_PROMPT_TEMPLATE = """নিচের বাংলা প্রবন্ধটি সংক্ষিপ্তভাবে সারাংশ করুন:

{text}

সারাংশ:"""

# ============================================================================
# STUDENT TRAINING CONFIG (LoRA + Distillation)
# ============================================================================
# LoRA
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training
STUDENT_MAX_INPUT_TOKENS = 850
STUDENT_MAX_OUTPUT_TOKENS = 256
STUDENT_BATCH_SIZE = 2                           # RTX 5080 16GB (was 4 for 5090)
STUDENT_GRADIENT_ACCUMULATION = 16               # Effective batch = 32 (2 * 16)
STUDENT_LEARNING_RATE = 2e-4
STUDENT_NUM_EPOCHS = 3                           # Total epochs across all runs
STUDENT_WARMUP_RATIO = 0.05
STUDENT_WEIGHT_DECAY = 0.01
STUDENT_MAX_GRAD_NORM = 1.0
STUDENT_BF16 = True
STUDENT_GRADIENT_CHECKPOINTING = True
STUDENT_SAVE_STEPS = 500
STUDENT_LOGGING_STEPS = 50

# ============================================================================
# EWAD (Entropy-Weighted Agreement-Aware Distillation) HYPERPARAMETERS
# ============================================================================
EWAD_TAU_W = 1.0       # Temperature for confidence-weighted softmax
EWAD_K = 5.0           # Sigmoid sharpness for agreement gate
EWAD_DELTA = 0.5       # Agreement threshold (tuned via validation)
EWAD_ENTROPY_EPS = 1e-8  # Epsilon for entropy clamping

# ============================================================================
# CPDP (Capacity-Proportional Divergence Preservation) HYPERPARAMETERS
# ============================================================================
CPDP_MU = 0.05          # Weight of CPDP loss in total loss
CPDP_EPS = 1e-8         # Epsilon for numerical stability

# ============================================================================
# COMBINED LOSS WEIGHTS
# ============================================================================
# L_total = L_EWAD + CPDP_MU * L_CPDP
# Where L_EWAD internally balances teacher KD vs gold CE via agreement gate

# ============================================================================
# EVALUATION CONFIG
# ============================================================================
EVAL_BATCH_SIZE = 4               # RTX 5080 16GB (beam search is memory-hungry)
EVAL_NUM_BEAMS = 4
EVAL_MAX_LENGTH = 256
EVAL_LENGTH_PENALTY = 1.2
QUICK_EVAL_SAMPLES = 200  # Subset for quick eval after each ablation

# ============================================================================
# EXPERIMENT ABLATION CONFIGS
# ============================================================================
EXPERIMENTS = {
    "baseline_no_distill": {
        "description": "Student alone (no distillation, fine-tune on gold labels)",
        "use_distillation": False,
        "use_ewad": False,
        "use_cpdp": False,
    },
    "single_teacher_32b": {
        "description": "32B → 3B only (standard KD)",
        "use_distillation": True,
        "teacher_weights": {"32b": 1.0, "14b": 0.0},
        "use_ewad": False,
        "use_cpdp": False,
    },
    
    "single_teacher_14b": {
        "description": "14B → 3B only (standard KD)",
        "use_distillation": True,
        "teacher_weights": {"32b": 0.0, "14b": 1.0},
        "use_ewad": False,
        "use_cpdp": False,
    },
    "fixed_weights": {
        "description": "Fixed α=0.7, β=0.3 dual-teacher KD",
        "use_distillation": True,
        "teacher_weights": {"32b": 0.7, "14b": 0.3},
        "use_ewad": False,
        "use_cpdp": False,
    },
    "confidence_only": {
        "description": "Confidence-weighted KD (no agreement gate)",
        "use_distillation": True,
        "use_ewad": "confidence_only",
        "use_cpdp": False,
    },
    "agreement_only": {
        "description": "Agreement-gated KD (no confidence weighting)",
        "use_distillation": True,
        "use_ewad": "agreement_only",
        "use_cpdp": False,
    },
    "ewad_full": {
        "description": "Full EWAD (confidence + agreement)",
        "use_distillation": True,
        "use_ewad": True,
        "use_cpdp": False,
    },
    "ewad_cpdp": {
        "description": "EWAD + CPDP (full system)",
        "use_distillation": True,
        "use_ewad": True,
        "use_cpdp": True,
    },
}

# ============================================================================
# TEST MODE OVERRIDES — lighter settings for quick pipeline sanity check
# ============================================================================
if TEST_MODE:
    MAX_SAMPLES = 1000                    # 1000 total → 900 train / 100 test
    STUDENT_NUM_EPOCHS = 1                # 1 epoch is enough to see if loss decreases
    STUDENT_SAVE_STEPS = 50               # Checkpoint more often (tiny dataset)
    STUDENT_LOGGING_STEPS = 10            # Log frequently
    STUDENT_GRADIENT_ACCUMULATION = 4     # Smaller effective batch (faster steps)
    TEACHER_BATCH_SIZE = 1                # Smaller batch to avoid OOM during test

    # Separate output dirs so test doesn't collide with full-run data
    TEACHER_32B_OUTPUTS = os.path.join(BASE_DIR, "teacher_outputs_test", "teacher_32b")
    TEACHER_14B_OUTPUTS = os.path.join(BASE_DIR, "teacher_outputs_test", "teacher_14b")
    STUDENT_OUTPUT_DIR = os.path.join(BASE_DIR, "student_outputs_test")
    EVAL_OUTPUT_DIR = os.path.join(BASE_DIR, "eval_results_test")

    print(f"\n*** TEST MODE ACTIVE — {MAX_SAMPLES} samples, {STUDENT_NUM_EPOCHS} epoch ***\n")

