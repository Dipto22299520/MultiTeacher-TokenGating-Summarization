"""
Chunk Processor — Solutions 1 & 2

Solution 1: Sentence-Aligned Chunking
  - Never chunk by tokens; chunk by sentences
  - Accumulate sentences until ~900 BPE tokens
  - Never break mid-sentence

Solution 2: Sliding Overlap (Context Carryover)
  - Overlap last N sentences between consecutive chunks
  - ~10-15% token overlap for discourse continuity

Uses the BanglaT5 tokenizer for accurate BPE token counting
(not whitespace word count, which underestimates Bangla BPE tokens).
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from bangla_sentence_splitter import split_sentences, count_bpe_tokens, get_bangla_tokenizer


@dataclass
class Chunk:
    """A single chunk of text with metadata."""
    text: str
    sentences: List[str]
    chunk_index: int
    total_chunks: int = 0  # filled after all chunks are created
    bpe_token_count: int = 0
    overlap_sentences: List[str] = field(default_factory=list)  # sentences from previous chunk
    
    def __post_init__(self):
        if self.bpe_token_count == 0:
            self.bpe_token_count = count_bpe_tokens(self.text)


class SentenceChunker:
    """
    Chunks long Bangla articles into model-sized pieces using sentence boundaries.
    
    Implements:
      - Solution 1: Sentence-aligned chunking (never splits mid-sentence)
      - Solution 2: Sliding overlap (carries over last N sentences)
    
    Args:
        max_tokens: Maximum BPE tokens per chunk (default 900, leaves room for
                    prefix "summarize: " and special tokens within 1024 limit)
        overlap_sentences: Number of sentences to carry over between chunks (default 3)
        min_chunk_tokens: Minimum tokens for a chunk to be valid (default 50)
                          Prevents tiny trailing chunks
    """
    
    def __init__(
        self,
        max_tokens: int = 900,
        overlap_sentences: int = 3,
        min_chunk_tokens: int = 50
    ):
        self.max_tokens = max_tokens
        self.overlap_sentences = overlap_sentences
        self.min_chunk_tokens = min_chunk_tokens
        self._tokenizer = None
    
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = get_bangla_tokenizer()
        return self._tokenizer
    
    def _count_tokens(self, text: str) -> int:
        """Count BPE tokens using the actual model tokenizer."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))
    
    def chunk_article(self, text: str, verbose: bool = False) -> List[Chunk]:
        """
        Split an article into sentence-aligned, overlapping chunks.
        
        Args:
            text: Full article text
            verbose: Enable detailed logging
            
        Returns:
            List of Chunk objects with metadata
        """
        if verbose:
            print(f"      [chunker] Splitting sentences...", flush=True)
        sentences = split_sentences(text)
        if verbose:
            print(f"      [chunker] Found {len(sentences)} sentences", flush=True)
        
        if not sentences:
            return []
        
        # Pre-compute BPE token count for EACH sentence (done once, O(n))
        if verbose:
            print(f"      [chunker] Counting tokens for {len(sentences)} sentences...", flush=True)
        sentence_tokens = count_bpe_tokens(sentences)  # batch call
        if verbose:
            print(f"      [chunker] Token counting done", flush=True)
        
        total_tokens = sum(sentence_tokens)
        if total_tokens <= self.max_tokens:
            if verbose:
                print(f"      [chunker] Article fits in 1 chunk", flush=True)
            chunk = Chunk(
                text=text,
                sentences=sentences,
                chunk_index=0,
                total_chunks=1,
                bpe_token_count=total_tokens,
                overlap_sentences=[]
            )
            return [chunk]
        
        # Step 2: Build chunks with sentence alignment + overlap
        if verbose:
            print(f"      [chunker] Building chunks...", flush=True)
        chunks = self._build_chunks(sentences, sentence_tokens, verbose=verbose)
        if verbose:
            print(f"      [chunker] Built {len(chunks)} chunks", flush=True)
        
        # Step 3: Set total_chunks on all chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _build_chunks(self, sentences: List[str], sentence_tokens: List[int], verbose: bool = False) -> List[Chunk]:
        """
        Build sentence-aligned chunks with sliding overlap.
        
        Uses pre-computed per-sentence token counts for speed (no re-tokenization).
        
        Algorithm:
          1. Start with empty chunk
          2. Add sentences one by one, tracking token sum
          3. When adding the next sentence would exceed max_tokens, finalize current chunk
          4. Start new chunk with overlap_sentences from previous chunk
          5. Repeat until all sentences consumed
        """
        chunks = []
        chunk_idx = 0
        
        # Current chunk state
        current_indices: List[int] = []   # indices into sentences[]
        current_tokens = 0
        overlap_indices: List[int] = []   # indices of overlap sentences
        
        i = 0
        if verbose:
            print(f"      [_build_chunks] Starting loop over {len(sentences)} sentences...", flush=True)
        
        max_iterations = len(sentences) * 100  # Safety limit: should never exceed this
        iteration_count = 0
        
        while i < len(sentences):
            iteration_count += 1
            if iteration_count > max_iterations:
                # Emergency break: infinite loop detected
                if verbose:
                    print(f"      [_build_chunks] EMERGENCY STOP: exceeded {max_iterations} iterations!", flush=True)
                # Force remaining sentences into final chunk
                if current_indices:
                    chunk = self._make_chunk_from_indices(
                        sentences, sentence_tokens, current_indices,
                        chunk_idx, overlap_indices, verbose=verbose
                    )
                    chunks.append(chunk)
                break
            
            if verbose and i % 5 == 0:
                print(f"      [_build_chunks] Processing sentence {i}/{len(sentences)}", flush=True)
            stok = sentence_tokens[i]
            
            # Handle extremely long single sentences (longer than max_tokens)
            if stok > self.max_tokens:
                if current_indices:
                    chunk = self._make_chunk_from_indices(
                        sentences, sentence_tokens, current_indices,
                        chunk_idx, overlap_indices, verbose=verbose
                    )
                    chunks.append(chunk)
                    chunk_idx += 1
                    overlap_indices = self._get_overlap_indices(current_indices)
                    current_indices = []
                    current_tokens = 0
                
                chunk = self._make_chunk_from_indices(
                    sentences, sentence_tokens, [i],
                    chunk_idx, overlap_indices, verbose=verbose
                )
                chunks.append(chunk)
                chunk_idx += 1
                overlap_indices = [i]
                i += 1
                continue
            
            # Would adding this sentence exceed the limit?
            separator_tokens = 1 if current_indices else 0
            projected_tokens = current_tokens + stok + separator_tokens
            
            if projected_tokens > self.max_tokens and current_indices:
                chunk = self._make_chunk_from_indices(
                    sentences, sentence_tokens, current_indices,
                    chunk_idx, overlap_indices, verbose=verbose
                )
                chunks.append(chunk)
                chunk_idx += 1
                
                overlap_indices = self._get_overlap_indices(current_indices)
                
                # Check: would overlap + current sentence exceed limit?
                # If so, reduce overlap to allow progress
                overlap_tokens = sum(sentence_tokens[j] for j in overlap_indices)
                if overlap_tokens + stok > self.max_tokens:
                    # Overlap is too large, reduce it
                    while overlap_indices and (sum(sentence_tokens[j] for j in overlap_indices) + stok > self.max_tokens):
                        overlap_indices.pop(0)  # Remove oldest overlap sentence
                    if verbose:
                        print(f"      [_build_chunks] Reduced overlap to {len(overlap_indices)} sentences to fit sentence {i}", flush=True)
                
                # Start new chunk WITH overlap sentences
                current_indices = list(overlap_indices)
                current_tokens = sum(sentence_tokens[j] for j in current_indices)
                
                # Don't advance i — re-process this sentence in the new chunk
                continue
            
            # Add sentence to current chunk
            current_indices.append(i)
            current_tokens += stok + separator_tokens
            i += 1
        
        # Finalize last chunk
        if current_indices:
            last_tokens = current_tokens
            new_only = [j for j in current_indices if j not in overlap_indices]
            
            if (last_tokens < self.min_chunk_tokens
                    and chunks and new_only):
                # Try merging with previous chunk
                prev_chunk = chunks[-1]
                new_sentences = [sentences[j] for j in new_only]
                new_tok = sum(sentence_tokens[j] for j in new_only)
                merged_tokens = prev_chunk.bpe_token_count + new_tok
                if merged_tokens <= self.max_tokens * 1.1:
                    prev_chunk.text += ' ' + ' '.join(new_sentences)
                    prev_chunk.sentences.extend(new_sentences)
                    prev_chunk.bpe_token_count = merged_tokens
                else:
                    chunk = self._make_chunk_from_indices(
                        sentences, sentence_tokens, current_indices,
                        chunk_idx, overlap_indices, verbose=verbose
                    )
                    chunks.append(chunk)
            elif new_only:  # has new content
                chunk = self._make_chunk_from_indices(
                    sentences, sentence_tokens, current_indices,
                    chunk_idx, overlap_indices, verbose=verbose
                )
                chunks.append(chunk)
        
        return chunks
    
    def _make_chunk_from_indices(
        self,
        sentences: List[str],
        sentence_tokens: List[int],
        indices: List[int],
        chunk_idx: int,
        overlap_indices: List[int],
        verbose: bool = False
    ) -> Chunk:
        """Create a Chunk from sentence indices (no re-tokenization)."""
        if verbose:
            print(f"      [_make_chunk] Creating chunk {chunk_idx} from {len(indices)} sentences", flush=True)
        sents = [sentences[j] for j in indices]
        text = ' '.join(sents)
        token_count = sum(sentence_tokens[j] for j in indices)
        if verbose:
            print(f"      [_make_chunk] Token count: {token_count}", flush=True)
        overlap_sents = [sentences[j] for j in overlap_indices] if overlap_indices else []
        if verbose:
            print(f"      [_make_chunk] Creating Chunk object...", flush=True)
        chunk = Chunk(
            text=text,
            sentences=sents,
            chunk_index=chunk_idx,
            bpe_token_count=token_count,
            overlap_sentences=overlap_sents
        )
        if verbose:
            print(f"      [_make_chunk] Done", flush=True)
        return chunk
    
    def _get_overlap_indices(self, indices: List[int]) -> List[int]:
        """Get the last N indices for overlap."""
        if self.overlap_sentences <= 0 or not indices:
            return []
        return indices[-self.overlap_sentences:]


def chunk_article_simple(text: str, max_tokens: int = 900, overlap: int = 3) -> List[Dict]:
    """
    Convenience function: chunk an article and return plain dicts.
    
    Returns:
        List of dicts with keys: text, chunk_index, total_chunks, 
        bpe_tokens, num_sentences, overlap_count
    """
    chunker = SentenceChunker(max_tokens=max_tokens, overlap_sentences=overlap)
    chunks = chunker.chunk_article(text)
    
    return [
        {
            'text': c.text,
            'chunk_index': c.chunk_index,
            'total_chunks': c.total_chunks,
            'bpe_tokens': c.bpe_token_count,
            'num_sentences': len(c.sentences),
            'overlap_count': len(c.overlap_sentences),
        }
        for c in chunks
    ]


# ============================================================================
# Self-test
# ============================================================================
if __name__ == "__main__":
    import json
    
    # Load a sample from gt_1000
    print("Loading sample from bangla_train_gt_1000.json...")
    with open("bangla_train_gt_1000.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Total articles in gt_1000: {len(data)}")
    
    # Test on first 3 articles
    chunker = SentenceChunker(max_tokens=900, overlap_sentences=3)
    
    for idx in range(min(3, len(data))):
        article = data[idx]
        text = article['text']
        summary = article['summary']
        
        total_tokens = count_bpe_tokens(text)
        chunks = chunker.chunk_article(text)
        
        print(f"\n{'='*60}")
        print(f"Article {idx+1}")
        print(f"  Original BPE tokens: {total_tokens}")
        print(f"  Summary: {summary[:100]}...")
        print(f"  Chunks created: {len(chunks)}")
        
        for c in chunks:
            print(f"\n  Chunk {c.chunk_index+1}/{c.total_chunks}:")
            print(f"    BPE tokens: {c.bpe_token_count}")
            print(f"    Sentences: {len(c.sentences)}")
            print(f"    Overlap sentences: {len(c.overlap_sentences)}")
            print(f"    Preview: {c.text[:80]}...")
