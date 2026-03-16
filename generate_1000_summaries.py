"""
Generate 1000 summaries for Multi-Judge evaluation.
Saves results to CSV with timing information.
"""

import json
import csv
import time
import sys
from datetime import datetime
from run_pipeline import ChunkedSummarizer

def main():
    # Load test data
    print("Loading test data...")
    with open("data_splits/test.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    total_available = len(test_data)
    num_samples = min(1000, total_available)
    print(f"Total test samples: {total_available}, generating: {num_samples}")
    
    # Initialize pipeline
    print("\nInitializing ChunkedSummarizer...")
    summarizer = ChunkedSummarizer()
    print("Models loaded.\n")
    
    # Output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"multi_judge_1000_summaries_{timestamp}.csv"
    
    # Generate summaries
    results = []
    total_start = time.time()
    
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "index", "article", "reference_summary", "generated_summary",
            "method", "num_chunks", "input_tokens", "time_seconds"
        ])
        
        for i in range(num_samples):
            sample = test_data[i]
            article = sample["text"]
            reference = sample["summary"]
            
            start = time.time()
            result = summarizer.summarize(article)
            elapsed = time.time() - start
            
            generated = result["summary"]
            method = result["method"]
            num_chunks = result["num_chunks"]
            input_tokens = result["input_tokens"]
            
            writer.writerow([
                i + 1, article, reference, generated,
                method, num_chunks, input_tokens, f"{elapsed:.2f}"
            ])
            csvfile.flush()
            
            # Progress
            if (i + 1) % 10 == 0 or i == 0:
                total_elapsed = time.time() - total_start
                avg = total_elapsed / (i + 1)
                remaining = avg * (num_samples - i - 1)
                print(
                    f"[{i+1}/{num_samples}] "
                    f"method={method} chunks={num_chunks} tokens={input_tokens} "
                    f"time={elapsed:.2f}s | "
                    f"avg={avg:.2f}s/article | "
                    f"ETA={remaining/60:.1f}min"
                )
            
            results.append({
                "method": method,
                "num_chunks": num_chunks,
                "input_tokens": input_tokens,
                "time": elapsed,
            })
    
    # Summary stats
    total_time = time.time() - total_start
    methods = {}
    for r in results:
        m = r["method"]
        if m not in methods:
            methods[m] = {"count": 0, "total_time": 0}
        methods[m]["count"] += 1
        methods[m]["total_time"] += r["time"]
    
    print("\n" + "=" * 60)
    print(f"DONE — {num_samples} summaries generated")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Average: {total_time/num_samples:.2f}s per article")
    print(f"Output: {csv_path}")
    print(f"\nMethod breakdown:")
    for m, stats in methods.items():
        avg_t = stats["total_time"] / stats["count"]
        print(f"  {m}: {stats['count']} articles, avg {avg_t:.2f}s each")
    print("=" * 60)


if __name__ == "__main__":
    main()
