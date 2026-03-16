"""
Memory-Aware Chunk Headers — Solution 3

Before each chunk (except the first), inject a short memory summary
that captures the gist of previous chunks. This gives the model
explicit context without relying solely on overlap.

Two modes:
  - Training: extractive memory (first sentence of each prior chunk)
  - Inference: model-generated memory (previous chunk's summary output)

Format injected before chunk content:
  [পূর্ববর্তী সারাংশ: <memory text>] <chunk content>
"""

from typing import List, Optional
from dataclasses import dataclass
from chunk_processor import Chunk, SentenceChunker


# Bangla prefix for memory header
MEMORY_PREFIX = "পূর্ববর্তী সারাংশ:"  # "Previous summary:"

# Maximum tokens allocated to the memory header
MAX_MEMORY_TOKENS = 100

# Minimum tokens that must remain for actual chunk content after memory header
MIN_CONTENT_TOKENS = 600


def _estimate_tokens(text: str) -> int:
    """Fast token estimate using word count * 2.5 (Bangla BPE ratio).
    
    Bangla text averages ~2.0-3.0 BPE tokens per whitespace word.
    We use 2.5 as a conservative estimate to stay within budget
    WITHOUT calling the tokenizer (which causes OOM at scale).
    """
    return int(len(text.split()) * 2.5)


@dataclass
class MemoryChunk:
    """A chunk with an optional memory header prepended."""
    original_chunk: Chunk
    memory_header: str              # The memory text (empty for first chunk)
    full_text: str                  # memory_header + original chunk text
    full_bpe_tokens: int
    chunk_index: int
    total_chunks: int
    
    @property
    def has_memory(self) -> bool:
        return bool(self.memory_header)


class MemoryHeaderInjector:
    """
    Injects memory summaries before each chunk to provide context
    from previous chunks.
    
    For training:
      - Uses extractive summarization: takes the first sentence of each
        prior chunk as a representative "memory" of what came before.
      - This is cheap, deterministic, and good enough to teach the model
        to leverage memory headers.
    
    For inference:
      - Pass the model's generated summary from each chunk as the memory
        for the next chunk. See inference_pipeline.py.
    
    Args:
        max_memory_tokens: Maximum BPE tokens for the memory header (default 100)
        min_content_tokens: Minimum tokens that must remain for chunk content (default 600)
    """
    
    def __init__(
        self,
        max_memory_tokens: int = MAX_MEMORY_TOKENS,
        min_content_tokens: int = MIN_CONTENT_TOKENS
    ):
        self.max_memory_tokens = max_memory_tokens
        self.min_content_tokens = min_content_tokens
    
    def inject_training_memory(self, chunks: List[Chunk]) -> List[MemoryChunk]:
        """
        Inject extractive memory headers for training.
        
        For chunk 0: no memory header
        For chunk N>0: memory = first sentence of each chunk 0..N-1,
                       truncated to max_memory_tokens
        
        Args:
            chunks: List of Chunk objects from SentenceChunker
            
        Returns:
            List of MemoryChunk objects with headers injected
        """
        if not chunks:
            return []
        
        memory_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk: no memory header
                mc = MemoryChunk(
                    original_chunk=chunk,
                    memory_header="",
                    full_text=chunk.text,
                    full_bpe_tokens=chunk.bpe_token_count,
                    chunk_index=chunk.chunk_index,
                    total_chunks=chunk.total_chunks
                )
            else:
                # Collect first sentence of each previous chunk
                memory_sentences = []
                for prev_chunk in chunks[:i]:
                    if prev_chunk.sentences:
                        # Skip overlap-only sentences to avoid repeating content
                        non_overlap = [
                            s for s in prev_chunk.sentences 
                            if s not in prev_chunk.overlap_sentences
                        ]
                        if non_overlap:
                            memory_sentences.append(non_overlap[0])
                        elif prev_chunk.sentences:
                            memory_sentences.append(prev_chunk.sentences[0])
                
                # Build memory header and truncate to token budget
                memory_text = self._build_memory_header(memory_sentences)
                
                # Combine with chunk content
                if memory_text:
                    full_text = f"[{MEMORY_PREFIX} {memory_text}] {chunk.text}"
                else:
                    full_text = chunk.text
                
                # Estimate tokens (fast) — exact count computed later if needed
                full_tokens = chunk.bpe_token_count + _estimate_tokens(memory_text) + 15
                
                mc = MemoryChunk(
                    original_chunk=chunk,
                    memory_header=memory_text,
                    full_text=full_text,
                    full_bpe_tokens=full_tokens,
                    chunk_index=chunk.chunk_index,
                    total_chunks=chunk.total_chunks
                )
            
            memory_chunks.append(mc)
        
        return memory_chunks
    
    def inject_inference_memory(
        self, 
        chunks: List[Chunk], 
        previous_summaries: List[str]
    ) -> List[MemoryChunk]:
        """
        Inject model-generated memory headers for inference.
        
        Uses the model's own summary outputs from previous chunks as the
        memory for subsequent chunks.
        
        Args:
            chunks: List of Chunk objects from SentenceChunker
            previous_summaries: List of generated summaries from chunks 0..N-1
                               (length should equal len(chunks), empty string for chunk 0)
        
        Returns:
            List of MemoryChunk objects with headers injected
        """
        if not chunks:
            return []
        
        memory_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0 or not previous_summaries or i > len(previous_summaries):
                # First chunk or no previous summaries
                mc = MemoryChunk(
                    original_chunk=chunk,
                    memory_header="",
                    full_text=chunk.text,
                    full_bpe_tokens=chunk.bpe_token_count,
                    chunk_index=chunk.chunk_index,
                    total_chunks=chunk.total_chunks
                )
            else:
                # Use all previous summaries as memory, truncated
                available_summaries = previous_summaries[:i]
                # Filter empty summaries
                available_summaries = [s for s in available_summaries if s.strip()]
                
                memory_text = self._build_memory_header(available_summaries)
                
                if memory_text:
                    full_text = f"[{MEMORY_PREFIX} {memory_text}] {chunk.text}"
                else:
                    full_text = chunk.text
                
                # Estimate tokens (fast)
                full_tokens = chunk.bpe_token_count + _estimate_tokens(memory_text) + 15
                
                mc = MemoryChunk(
                    original_chunk=chunk,
                    memory_header=memory_text,
                    full_text=full_text,
                    full_bpe_tokens=full_tokens,
                    chunk_index=chunk.chunk_index,
                    total_chunks=chunk.total_chunks
                )
            
            memory_chunks.append(mc)
        
        return memory_chunks
    
    def _build_memory_header(self, sentences: List[str]) -> str:
        """
        Build a memory header from a list of sentences, respecting token budget.
        
        Uses fast word-count estimation instead of tokenizer calls.
        """
        if not sentences:
            return ""
        
        selected = []
        current_est = 0
        
        for sentence in sentences:
            est = _estimate_tokens(sentence)
            
            if current_est + est > self.max_memory_tokens:
                if not selected:
                    # Must include at least something — truncate by words
                    words = sentence.split()
                    # Take roughly max_memory_tokens / 2.5 words
                    max_words = max(1, int(self.max_memory_tokens / 2.5))
                    selected.append(' '.join(words[:max_words]))
                break
            
            selected.append(sentence)
            current_est += est
        
        return ' '.join(selected)


def process_article_with_memory(
    text: str,
    max_tokens: int = 900,
    overlap_sentences: int = 3,
    max_memory_tokens: int = 100,
    verbose: bool = False
) -> List[MemoryChunk]:
    """
    Full pipeline: sentence chunk → overlap → memory headers.
    
    Adjusts chunk size to leave room for memory headers.
    
    Args:
        text: Full article text
        max_tokens: Max BPE tokens per chunk (before memory header)
        overlap_sentences: Number of overlap sentences between chunks
        max_memory_tokens: Max tokens for memory header
    
    Returns:
        List of MemoryChunk objects
    """
    # Reduce chunk size to leave room for memory header + prefix
    # The "[পূর্ববর্তী সারাংশ: ...]" wrapper adds ~10-15 tokens overhead
    effective_max = max_tokens - max_memory_tokens - 20  # 20 for bracket/prefix tokens
    effective_max = max(effective_max, 500)  # Safety floor
    
    if verbose:
        print(f"      [memory_header] Starting chunker...", flush=True)
    chunker = SentenceChunker(
        max_tokens=effective_max,
        overlap_sentences=overlap_sentences
    )
    if verbose:
        print(f"      [memory_header] Calling chunk_article...", flush=True)
    chunks = chunker.chunk_article(text, verbose=verbose)
    if verbose:
        print(f"      [memory_header] Got {len(chunks)} chunks, injecting memory...", flush=True)
    
    # If only 1 chunk, no memory needed — but restore to full max_tokens chunking
    if len(chunks) <= 1:
        # Re-chunk with full budget since no memory needed
        if verbose:
            print(f"      [memory_header] Only 1 chunk, re-chunking with full budget...", flush=True)
        chunker_full = SentenceChunker(
            max_tokens=max_tokens,
            overlap_sentences=overlap_sentences
        )
        chunks = chunker_full.chunk_article(text, verbose=verbose)
        if len(chunks) <= 1:
            # Still fits in one chunk
            injector = MemoryHeaderInjector(max_memory_tokens=max_memory_tokens)
            return injector.inject_training_memory(chunks)
    
    injector = MemoryHeaderInjector(max_memory_tokens=max_memory_tokens)
    return injector.inject_training_memory(chunks)


# ============================================================================
# Self-test
# ============================================================================
if __name__ == "__main__":
    import json
    
    print("Loading sample from bangla_train_gt_1000.json...")
    with open("bangla_train_gt_1000.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Test on first article
    article = data[0]
    text = article['text']
    summary = article['summary']
    
    print(f"\nOriginal article word count: {len(text.split())}")
    print(f"Estimated BPE tokens: {_estimate_tokens(text)}")
    print(f"Summary: {summary[:100]}...")
    
    memory_chunks = process_article_with_memory(text)
    
    print(f"\nCreated {len(memory_chunks)} memory-enhanced chunks:")
    for mc in memory_chunks:
        print(f"\n  Chunk {mc.chunk_index + 1}/{mc.total_chunks}:")
        print(f"    Has memory header: {mc.has_memory}")
        if mc.has_memory:
            print(f"    Memory: {mc.memory_header[:80]}...")
        print(f"    Full text tokens: {mc.full_bpe_tokens}")
        print(f"    Content preview: {mc.original_chunk.text[:80]}...")
