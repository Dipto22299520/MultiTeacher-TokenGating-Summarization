"""
MapReduce Hierarchical Summarization Pipeline

For inference on long (>1000 token) Bangla articles:

  MAP phase:
    1. Sentence-chunk the article (Solutions 1+2: sentence-aligned + overlap)
    2. Inject memory headers (Solution 3: extractive or model-generated)
    3. Summarize each chunk independently with the fine-tuned model

  REDUCE phase:
    4. Concatenate chunk summaries
    5. If the concatenation fits in 1024 tokens → summarize again for coherence
    6. If it doesn't fit → recursively chunk and summarize (rare for <10 chunks)

Why MapReduce over simple concatenation:
  - Overlapping chunks create redundant content in chunk summaries
  - A second summarization pass deduplicates and produces coherent output
  - Produces a single, fluent summary instead of a list of chunk summaries

Usage:
    from inference_pipeline import HierarchicalSummarizer
    
    summarizer = HierarchicalSummarizer("path/to/final_model")
    summary = summarizer.summarize("long bangla article text...")
"""

import os
import torch
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from chunk_processor import SentenceChunker, Chunk
from memory_header import MemoryHeaderInjector, MEMORY_PREFIX
from bangla_sentence_splitter import count_bpe_tokens


@dataclass
class SummarizationResult:
    """Result from the hierarchical summarization pipeline."""
    final_summary: str
    chunk_summaries: List[str]
    num_chunks: int
    num_reduce_passes: int
    total_input_tokens: int
    final_summary_tokens: int
    
    def __repr__(self):
        return (
            f"SummarizationResult(\n"
            f"  chunks={self.num_chunks}, "
            f"  reduce_passes={self.num_reduce_passes}, "
            f"  input_tokens={self.total_input_tokens}, "
            f"  output_tokens={self.final_summary_tokens}\n"
            f"  summary={self.final_summary[:100]}...\n"
            f")"
        )


class HierarchicalSummarizer:
    """
    Production inference pipeline for long Bangla article summarization.
    
    Handles both short articles (≤1024 tokens, single-pass) and long articles
    (>1024 tokens, MapReduce chunked summarization).
    
    Args:
        model_path: Path to the fine-tuned model directory
        max_chunk_tokens: Max BPE tokens per chunk (default 900)
        overlap_sentences: Number of overlap sentences (default 3)
        max_memory_tokens: Max tokens for memory headers (default 100)
        use_memory_headers: Enable Solution 3 memory headers (default True)
        use_model_generated_memory: Use model output as memory for next chunk
                                     (True=model-generated, False=extractive)
        device: 'cuda' or 'cpu' (default: auto-detect)
        
    Generation kwargs:
        num_beams, max_length, length_penalty, no_repeat_ngram_size,
        repetition_penalty, early_stopping
    """
    
    def __init__(
        self,
        model_path: str,
        max_chunk_tokens: int = 900,
        overlap_sentences: int = 3,
        max_memory_tokens: int = 100,
        use_memory_headers: bool = True,
        use_model_generated_memory: bool = True,
        device: Optional[str] = None,
        # Generation parameters (matching updated training config)
        num_beams: int = 6,
        max_length: int = 256,
        length_penalty: float = 1.2,
        no_repeat_ngram_size: int = 3,
        repetition_penalty: float = 1.15,
        early_stopping: bool = True,
    ):
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_sentences = overlap_sentences
        self.max_memory_tokens = max_memory_tokens
        self.use_memory_headers = use_memory_headers
        self.use_model_generated_memory = use_model_generated_memory
        
        # Device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Generation parameters
        self.gen_kwargs = {
            "num_beams": num_beams,
            "max_length": max_length,
            "min_length": 64,
            "length_penalty": length_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "repetition_penalty": repetition_penalty,
            "early_stopping": early_stopping,
            "diversity_penalty": 0.3,
        }
        
        # Load model and tokenizer
        print(f"Loading model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        
        # Check if this is a ChunkAwareT5 model
        bias_path = os.path.join(model_path, "chunk_attention_bias.pt")
        if os.path.exists(bias_path):
            from attention_bias import ChunkAwareT5
            self.model = ChunkAwareT5.from_saved(model_path, self.tokenizer)
            self._is_chunk_aware = True
            print("  ✓ Loaded ChunkAwareT5 model with attention bias")
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            self._is_chunk_aware = False
            print("  ✓ Loaded standard T5 model")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize chunker and memory
        self.chunker = SentenceChunker(
            max_tokens=max_chunk_tokens,
            overlap_sentences=overlap_sentences
        )
        
        self.memory_injector = MemoryHeaderInjector(
            max_memory_tokens=max_memory_tokens
        )
        
        print(f"  ✓ Device: {self.device}")
        print(f"  ✓ Ready for inference")
    
    @torch.no_grad()
    def summarize(self, text: str) -> SummarizationResult:
        """
        Summarize a Bangla article (any length).
        
        For short articles (≤1024 tokens): single-pass summarization.
        For long articles (>1024 tokens): MapReduce pipeline.
        
        Args:
            text: Full article text
            
        Returns:
            SummarizationResult with final summary and metadata
        """
        total_tokens = count_bpe_tokens(text)
        
        # Single-pass for short articles
        if total_tokens <= self.max_chunk_tokens:
            summary = self._generate_summary(text)
            return SummarizationResult(
                final_summary=summary,
                chunk_summaries=[summary],
                num_chunks=1,
                num_reduce_passes=0,
                total_input_tokens=total_tokens,
                final_summary_tokens=count_bpe_tokens(summary)
            )
        
        # MapReduce for long articles
        return self._map_reduce(text, total_tokens)
    
    def _map_reduce(self, text: str, total_tokens: int) -> SummarizationResult:
        """
        MAP phase: Chunk → Summarize each chunk
        REDUCE phase: Merge chunk summaries → Final summary
        """
        # === MAP PHASE ===
        chunks = self.chunker.chunk_article(text)
        
        chunk_summaries = []
        previous_summaries = []
        
        for i, chunk in enumerate(chunks):
            # Inject memory header
            if self.use_memory_headers and i > 0:
                if self.use_model_generated_memory and previous_summaries:
                    # Use model's own output from previous chunks as memory
                    memory_chunks = self.memory_injector.inject_inference_memory(
                        [chunk], 
                        previous_summaries
                    )
                    chunk_text = memory_chunks[0].full_text
                else:
                    # Use extractive memory (first sentence of previous chunks)
                    memory_chunks = self.memory_injector.inject_training_memory(
                        chunks[:i+1]
                    )
                    chunk_text = memory_chunks[-1].full_text
            else:
                chunk_text = chunk.text
            
            # Generate summary for this chunk (detailed/extractive for MAP phase)
            summary = self._generate_summary(chunk_text, is_reduce_phase=False)
            chunk_summaries.append(summary)
            previous_summaries.append(summary)
        
        # === REDUCE PHASE ===
        final_summary, num_reduce_passes = self._reduce(chunk_summaries)
        
        return SummarizationResult(
            final_summary=final_summary,
            chunk_summaries=chunk_summaries,
            num_chunks=len(chunks),
            num_reduce_passes=num_reduce_passes,
            total_input_tokens=total_tokens,
            final_summary_tokens=count_bpe_tokens(final_summary)
        )
    
    def _reduce(self, chunk_summaries: List[str]) -> Tuple[str, int]:
        """
        Reduce chunk summaries into a single coherent summary.
        
        If concatenated summaries fit in 1024 tokens → one-pass reduce.
        Otherwise → recursive chunking (rare).
        
        Returns:
            (final_summary, num_reduce_passes)
        """
        if len(chunk_summaries) == 1:
            return chunk_summaries[0], 0
        
        # Concatenate all chunk summaries
        merged = ' '.join(chunk_summaries)
        merged_tokens = count_bpe_tokens(merged)
        
        num_passes = 0
        
        # Recursive reduce if too long
        while merged_tokens > self.max_chunk_tokens:
            num_passes += 1
            
            # Re-chunk the merged summaries
            sub_chunks = self.chunker.chunk_article(merged)
            
            if len(sub_chunks) <= 1:
                # Can't split further, just truncate
                break
            
            # Summarize each sub-chunk
            sub_summaries = []
            for sc in sub_chunks:
                sub_summary = self._generate_summary(sc.text, is_reduce_phase=True)
                sub_summaries.append(sub_summary)
            
            merged = ' '.join(sub_summaries)
            merged_tokens = count_bpe_tokens(merged)
        
        # Final reduce pass: summarize for coherence & thematic focus
        num_passes += 1
        final_summary = self._generate_summary(merged, is_reduce_phase=True)
        
        # Post-process: remove repetitive sentences
        final_summary = self._remove_repetitions(final_summary)
        
        return final_summary, num_passes
    
    def _generate_summary(self, text: str, is_reduce_phase: bool = False) -> str:
        """
        Generate a summary for a single chunk of text.
        
        Args:
            text: Input text to summarize
            is_reduce_phase: If True, use abstractive/thematic prompt
                            If False, use detailed/extractive prompt
        """
        if is_reduce_phase:
            # REDUCE: Encourage abstractive, thematic, high-level summary
            input_text = f"summarize bangla news: write main themes and key points from: {text}"
        else:
            # MAP: Standard detailed summary of chunk
            input_text = f"summarize bangla news: {text}"
        
        inputs = self.tokenizer(
            input_text,
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        if self._is_chunk_aware:
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **self.gen_kwargs
            )
        else:
            outputs = self.model.generate(
                **inputs,
                **self.gen_kwargs
            )
        
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary.strip()
    
    def _remove_repetitions(self, text: str) -> str:
        """
        Remove repetitive sentences from generated summary.
        
        Uses simple sentence-level deduplication based on exact match
        and high word-overlap detection.
        """
        sentences = [s.strip() for s in text.split('।') if s.strip()]
        
        if len(sentences) <= 1:
            return text
        
        # Remove exact duplicates
        unique_sentences = []
        seen = set()
        
        for sent in sentences:
            # Normalize for comparison
            normalized = ' '.join(sent.split())
            
            # Skip if exact duplicate
            if normalized in seen:
                continue
            
            # Skip if high word-level overlap with existing sentences (>80%)
            is_repetitive = False
            sent_words = set(sent.split())
            
            for existing in unique_sentences:
                existing_words = set(existing.split())
                if not sent_words or not existing_words:
                    continue
                    
                # Jaccard similarity
                intersection = len(sent_words & existing_words)
                union = len(sent_words | existing_words)
                similarity = intersection / union if union > 0 else 0
                
                if similarity > 0.8:
                    is_repetitive = True
                    break
            
            if not is_repetitive:
                unique_sentences.append(sent)
                seen.add(normalized)
        
        # Reconstruct with sentence boundary marker
        return '।'.join(unique_sentences) + ('।' if text.endswith('।') else '')
    
    def summarize_batch(self, texts: List[str]) -> List[SummarizationResult]:
        """Summarize a batch of articles."""
        return [self.summarize(text) for text in texts]


# ============================================================================
# Standalone usage
# ============================================================================
if __name__ == "__main__":
    import json
    import sys
    
    # Default model path — update after training
    MODEL_PATH = None
    
    # Try to find latest trained model
    import glob
    model_dirs = sorted(glob.glob("./banglaT5_chunked_*/final_model"))
    if model_dirs:
        MODEL_PATH = model_dirs[-1]
    else:
        # Fallback to the lte_1000 model for testing the pipeline
        model_dirs = sorted(glob.glob("./banglaT5_production_*/final_model"))
        if model_dirs:
            MODEL_PATH = model_dirs[-1]
    
    if MODEL_PATH is None:
        print("❌ No trained model found. Train first with train_bangla_chunked.py")
        sys.exit(1)
    
    print(f"Using model: {MODEL_PATH}")
    
    # Load test data
    with open("bangla_train_gt_1000.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Initialize summarizer
    summarizer = HierarchicalSummarizer(
        model_path=MODEL_PATH,
        max_chunk_tokens=900,
        overlap_sentences=3,
        max_memory_tokens=100,
        use_memory_headers=True,
        use_model_generated_memory=True
    )
    
    # Test on first 3 articles
    print("\n" + "=" * 60)
    print("INFERENCE PIPELINE TEST")
    print("=" * 60)
    
    for i in range(min(3, len(data))):
        article = data[i]
        text = article['text']
        reference = article['summary']
        
        print(f"\n{'='*40} Article {i+1} {'='*40}")
        print(f"Input tokens: {count_bpe_tokens(text)}")
        print(f"Reference: {reference[:150]}...")
        
        result = summarizer.summarize(text)
        
        print(f"\n{result}")
        print(f"Final summary: {result.final_summary}")
        print(f"\nChunk summaries:")
        for j, cs in enumerate(result.chunk_summaries):
            print(f"  Chunk {j+1}: {cs[:100]}...")
