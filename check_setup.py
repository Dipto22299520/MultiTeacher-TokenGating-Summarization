"""
Pre-Training Validation Script

Run this before training to ensure everything is properly configured.

Usage:
    python check_setup.py
"""

import os
import sys
import json
import torch
from pathlib import Path


def check_file(filepath, description, required=True):
    """Check if a file exists."""
    exists = os.path.exists(filepath)
    symbol = "✅" if exists else ("❌" if required else "⚠️")
    print(f"{symbol} {description}: {filepath}")
    
    if not exists and required:
        print(f"   ERROR: Required file not found!")
        return False
    
    if exists:
        size = os.path.getsize(filepath)
        size_mb = size / (1024 * 1024)
        print(f"   Size: {size_mb:.1f} MB")
    
    return True


def check_dataset(filepath):
    """Validate dataset format."""
    print(f"\n{'='*80}")
    print("DATASET VALIDATION")
    print(f"{'='*80}")
    
    if not os.path.exists(filepath):
        print(f"❌ Dataset not found: {filepath}")
        return False
    
    print(f"✅ Dataset found: {filepath}")
    
    # Check file size
    size = os.path.getsize(filepath)
    size_mb = size / (1024 * 1024)
    print(f"   Size: {size_mb:.1f} MB")
    
    # Load and validate samples
    print("\n   Loading samples...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print("   ❌ Error: Dataset is not a JSON array")
            return False
        
        samples = data[:100]  # Check first 100
        total_samples = len(data)
        
        print(f"   ✅ Total samples in dataset: {total_samples:,}")
        print(f"   ✅ Loaded {len(samples)} samples for validation")
        
        # Check required fields
        valid_count = 0
        for i, sample in enumerate(samples[:10]):
            if 'text' in sample and 'summary' in sample:
                text = str(sample['text']).strip()
                summary = str(sample['summary']).strip()
                
                if len(text) > 50 and len(summary) > 10:
                    valid_count += 1
        
        print(f"   ✅ Valid samples: {valid_count}/10 checked")
        
        if valid_count < 5:
            print(f"   ⚠️  Warning: Many samples seem invalid")
            return False
        
        # Show sample
        print("\n   Sample data (first entry):")
        sample = samples[0]
        text = str(sample.get('text', ''))[:200]
        summary = str(sample.get('summary', ''))[:100]
        print(f"   Text: {text}...")
        print(f"   Summary: {summary}...")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error loading dataset: {e}")
        return False


def check_gpu():
    """Check GPU availability."""
    print(f"\n{'='*80}")
    print("GPU CHECK")
    print(f"{'='*80}")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA available")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        
        # Check memory
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   Total memory: {total_mem:.1f} GB")
        
        # Free memory
        torch.cuda.empty_cache()
        free_mem = torch.cuda.memory_reserved(0) / (1024**3)
        print(f"   Reserved memory: {free_mem:.1f} GB")
        
        if total_mem < 8:
            print(f"   ⚠️  Warning: GPU memory < 8GB, may need to reduce batch size")
        
        return True
    else:
        print(f"❌ CUDA not available")
        print(f"   PyTorch will use CPU (very slow for training)")
        print(f"   Make sure NVIDIA drivers and CUDA are installed")
        return False


def check_dependencies():
    """Check required Python packages."""
    print(f"\n{'='*80}")
    print("DEPENDENCY CHECK")
    print(f"{'='*80}")
    
    required = [
        ('torch', 'PyTorch'),
        ('transformers', 'HuggingFace Transformers'),
        ('datasets', 'HuggingFace Datasets'),
        ('rouge_score', 'ROUGE Score'),
        ('numpy', 'NumPy'),
        ('sklearn', 'Scikit-learn'),
    ]
    
    optional = [
        ('bert_score', 'BERTScore (recommended)'),
        ('tensorboard', 'TensorBoard (recommended)'),
    ]
    
    all_good = True
    
    print("\nRequired packages:")
    for module, name in required:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - MISSING!")
            print(f"   Install: pip install {module}")
            all_good = False
    
    print("\nOptional packages:")
    for module, name in optional:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"⚠️  {name} - not installed (recommended)")
            print(f"   Install: pip install {module}")
    
    return all_good


def check_disk_space():
    """Check available disk space."""
    print(f"\n{'='*80}")
    print("DISK SPACE CHECK")
    print(f"{'='*80}")
    
    # Get disk usage for current directory
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        
        free_gb = free / (1024**3)
        total_gb = total / (1024**3)
        
        print(f"   Total: {total_gb:.1f} GB")
        print(f"   Free: {free_gb:.1f} GB")
        
        if free_gb < 10:
            print(f"   ⚠️  Warning: Less than 10GB free")
            print(f"   Training will generate model checkpoints (~5-10GB)")
            return False
        else:
            print(f"   ✅ Sufficient space available")
            return True
            
    except Exception as e:
        print(f"   ⚠️  Could not check disk space: {e}")
        return True


def estimate_training_time():
    """Estimate training time."""
    print(f"\n{'='*80}")
    print("TRAINING TIME ESTIMATE")
    print(f"{'='*80}")
    
    # Rough estimates for RTX 5080 (16GB VRAM)
    print("\nEstimated times (RTX 5080, batch_size=4, grad_accum=8):")
    print("   Full Document Training (79k samples, 20 epochs):")
    print("   • Optimistic: 4-5 hours")
    print("   • Realistic: 5-7 hours")
    print("   • Conservative: 7-10 hours")
    
    print("\n   Reduce Task Training (79k samples, 15 epochs):")
    print("   • Optimistic: 3-4 hours")
    print("   • Realistic: 4-6 hours")
    print("   • Conservative: 6-8 hours")
    
    print("\n   Factors affecting speed:")
    print("   • GPU utilization (check nvidia-smi)")
    print("   • Data loading speed (SSD vs HDD)")
    print("   • System background tasks")
    
    print("\n   💡 Tip: Monitor with TensorBoard:")
    print("      tensorboard --logdir=./banglaT5_full_doc_XXXXXX/logs")


def main():
    """Run all checks."""
    print(f"\n{'='*80}")
    print("PRE-TRAINING SETUP VALIDATION")
    print(f"{'='*80}")
    print("This script checks if your environment is ready for training.\n")
    
    checks = []
    
    # Check files
    print(f"\n{'='*80}")
    print("FILE CHECK")
    print(f"{'='*80}")
    checks.append(check_file("bangla_train_combined.json", "Training dataset", required=True))
    checks.append(check_file("train_full_document.py", "Training script", required=True))
    checks.append(check_file("train_reduce_task.py", "Reduce task script", required=True))
    
    # Check dataset
    checks.append(check_dataset("bangla_train_combined.json"))
    
    # Check dependencies
    checks.append(check_dependencies())
    
    # Check GPU
    gpu_available = check_gpu()
    checks.append(gpu_available)
    
    # Check disk space
    checks.append(check_disk_space())
    
    # Estimate time
    estimate_training_time()
    
    # Summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")
    
    if all(checks):
        print("\n🎉 ALL CHECKS PASSED!")
        print("\n✅ Your environment is ready for training!")
        print("\nNext steps:")
        print("   1. Run: python train_full_document.py")
        print("   2. Monitor with TensorBoard")
        print("   3. Wait for training to complete (4-6 hours)")
        print("   4. Evaluate results with compare_models.py")
        
        return 0
    else:
        print("\n⚠️  SOME CHECKS FAILED")
        print("\nPlease fix the issues above before training.")
        print("Common fixes:")
        print("   • Install missing packages: pip install -r requirements.txt")
        print("   • Check CUDA installation: nvidia-smi")
        print("   • Verify dataset file exists and is valid")
        
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
