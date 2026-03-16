"""
Quick evaluation script to check current checkpoint quality
"""
import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Configuration
CHECKPOINT_DIR = "./banglaT5_production_20260210_131619/checkpoint-1500"
DATA_FILE = "bangla_train_lte_1000.json"
NUM_SAMPLES = 5

# Generation settings
NUM_BEAMS = 4
MAX_LENGTH = 192
PREFIX = "summarize: "

def main():
    print("=" * 80)
    print("QUICK CHECKPOINT EVALUATION")
    print("=" * 80)
    print(f"\nLoading checkpoint: {CHECKPOINT_DIR}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT_DIR)
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    print(f"Device: {device}")
    print(f"Model loaded successfully\n")
    
    # Load data
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get random samples
    import random
    random.seed(42)
    samples = random.sample(data, NUM_SAMPLES)
    
    print("=" * 80)
    print("SAMPLE PREDICTIONS")
    print("=" * 80)
    
    for i, sample in enumerate(samples, 1):
        print(f"\n{'='*40} Sample {i} {'='*40}")
        
        # Original text
        original_text = sample['text']
        reference_summary = sample['summary']
        
        print(f"\n📄 Original Text (first 300 chars):")
        print(f"   {original_text[:300]}...")
        
        print(f"\n🎯 Reference Summary:")
        print(f"   {reference_summary}")
        
        # Generate prediction
        input_text = PREFIX + original_text
        inputs = tokenizer(
            input_text,
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=MAX_LENGTH,
                num_beams=NUM_BEAMS,
                length_penalty=1.0,
                no_repeat_ngram_size=2,
                repetition_penalty=1.2,
                early_stopping=True
            )
        
        generated_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"\n🤖 Generated Summary:")
        print(f"   {generated_summary}")
        
        # Quick quality check
        if not generated_summary or len(generated_summary) < 10:
            print(f"\n⚠️  WARNING: Generated summary is too short or empty!")
        elif generated_summary == original_text[:len(generated_summary)]:
            print(f"\n⚠️  WARNING: Model is copying input text!")
        else:
            print(f"\n✓ Length: {len(generated_summary)} chars (reference: {len(reference_summary)} chars)")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
