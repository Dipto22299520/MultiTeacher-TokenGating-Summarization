# Sample 5 predictions to see quality
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("./banglaT5_full_doc_20260214_224524/checkpoint-2000", use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained("./banglaT5_full_doc_20260214_224524/checkpoint-2000")

with open("data_splits/test.json", 'r', encoding='utf-8') as f:
    test_data = json.load(f)

for i in range(3):
    text = "summarize bangla news: " + test_data[i]['text'][:500]
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(**inputs, max_length=256, num_beams=5)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"\n{'='*80}")
    print(f"SAMPLE {i+1}")
    print(f"{'='*80}")
    print(f"Reference: {test_data[i]['summary']}")
    print(f"Generated: {summary}")