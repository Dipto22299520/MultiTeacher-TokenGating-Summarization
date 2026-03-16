"""
Sample predictions to understand why word ROUGE is low but semantic similarity is high
"""
import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def fix_bangla_for_tokenizer(text):
    text = text.replace('\u09DF', '\u09AF\u09BC')
    text = text.replace('\u09DC', '\u09A1\u09BC')
    text = text.replace('\u09DD', '\u09A2\u09BC')
    text = text.replace('\u200c', '').replace('\u200d', '')
    return text

CHECKPOINT = "./banglaT5_full_doc_20260215_123349/checkpoint-7000"
print(f"Loading model from {CHECKPOINT}...")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT)

if torch.cuda.is_available():
    model = model.cuda()
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")

print("\nLoading test data...")
with open("data_splits/test.json", 'r', encoding='utf-8') as f:
    test_data = json.load(f)

print("\nGenerating 5 sample predictions...\n")

# Open file for saving all outputs
output_file = "sample_predictions_output.txt"
with open(output_file, 'w', encoding='utf-8') as f_out:
    f_out.write("=" * 80 + "\n")
    f_out.write("BANGLA SUMMARIZATION SAMPLE PREDICTIONS\n")
    f_out.write("=" * 80 + "\n\n")
    f_out.write("⚠️ If you see broken Bangla text in the console, this file has the correct output.\n")
    f_out.write("Open this file in VS Code or Notepad to view properly rendered Bangla text.\n\n")
    
    for i in range(10):
        text = "summarize bangla news: " + fix_bangla_for_tokenizer(test_data[i]['text'])
        inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        outputs = model.generate(
            **inputs, 
            max_length=256, 
            min_length=64,
            num_beams=5,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Console output (may have rendering issues)
        print("="*80)
        print(f"SAMPLE {i+1}")
        print("="*80)
        print(f"\n📰 ARTICLE (first 300 chars):")
        print(test_data[i]['text'][:300] + "...")
        print(f"\n✅ REFERENCE SUMMARY:")
        print(test_data[i]['summary'])
        print(f"\n🤖 GENERATED SUMMARY:")
        print(summary)
        print(f"\n📊 Lengths: Reference={len(test_data[i]['summary'].split())} words, Generated={len(summary.split())} words")
        print()
        
        # File output (properly encoded)
        f_out.write("=" * 80 + "\n")
        f_out.write(f"SAMPLE {i+1}\n")
        f_out.write("=" * 80 + "\n\n")
        f_out.write(f"📰 ARTICLE (first 300 chars):\n{test_data[i]['text'][:300]}...\n\n")
        f_out.write(f"✅ REFERENCE SUMMARY:\n{test_data[i]['summary']}\n\n")
        f_out.write(f"🤖 GENERATED SUMMARY:\n{summary}\n\n")
        f_out.write(f"📊 Lengths: Reference={len(test_data[i]['summary'].split())} words, Generated={len(summary.split())} words\n\n")

print(f"\n✅ All predictions saved to: {output_file}")
print("⚠️  If Bangla text looks broken in console, open the file in VS Code to see proper text.")
