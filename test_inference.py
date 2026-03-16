import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load the model and tokenizer
model_path = "mt5_teacher_mt5-base_20260208_124334/checkpoint-10000"
print(f"Loading model from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()
print(f"Model loaded on {device}\n")

# Load test data
print("Loading test data...")
test_df = pd.read_csv("data/test.csv")
print(f"Test data loaded: {len(test_df)} samples\n")

# Select random samples to test
num_samples = 5
print(f"Generating summaries for {num_samples} random samples...\n")
samples = test_df.sample(n=num_samples, random_state=42)

print("="*100)
for idx, row in enumerate(samples.itertuples(), 1):
    text = row.text
    reference_summary = row.summary
    category = row.category
    
    # Prepare input
    input_text = f"summarize: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate summary
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=150,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
    
    generated_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Display results
    print(f"\nSample {idx}")
    print(f"Category: {category}")
    print(f"\nInput Text (first 300 chars):\n{text[:300]}...")
    print(f"\nReference Summary:\n{reference_summary}")
    print(f"\nGenerated Summary:\n{generated_summary}")
    print("="*100)

print("\n✓ Inference completed!")
