"""
Bangla Sentence Segmenter — Rule-based

Splits Bangla text into sentences using:
  - Bangla danda (।)
  - Double danda (॥)
  - Question mark (?)
  - Exclamation mark (!)
  - Newlines (paragraph boundaries)

Edge-case handling:
  - Quoted speech: don't split on terminators inside quotes
  - Numbers with periods: 5.3, 10.25 etc.
  - Abbreviations: ড., প্রফ., etc.
  - Empty sentences filtered out
"""

import re
from typing import List, Union


# Common Bangla abbreviations that end with a period (English-style)
# These should NOT be treated as sentence boundaries
BANGLA_ABBREVIATIONS = {
    'ড.', 'ডা.', 'প্রফ.', 'মি.', 'মিসেস.', 'মিস.', 'জনাব.',
    'মো.', 'সৈ.', 'হযরত.', 'আ.', 'স.', 'রা.',
    'নং.', 'পৃ.', 'সা.', 'কি.মি.', 'মি.গ্রা.',
    'ইং.', 'বি.', 'এম.', 'পি.এইচ.ডি.', 'এ.', 'সি.',
}

# Regex for Bangla sentence terminators
# ।  = U+0964 (Devanagari Danda, used in Bangla)
# ॥  = U+0965 (Devanagari Double Danda)
_SENTENCE_TERMINATORS = r'[।॥?!]'

# Regex: digit.digit should NOT be a sentence boundary
_DECIMAL_PATTERN = re.compile(r'\d\.\d')

# Regex for the actual split: terminator followed by whitespace or end
_SPLIT_PATTERN = re.compile(
    r'('           # capture the terminator + trailing space
    r'[।॥]'       # Bangla danda / double danda
    r'|[?!]'      # question / exclamation
    r')'
    r'(?=\s|$)'   # followed by whitespace or end-of-string
)


def _is_inside_quotes(text: str, pos: int) -> bool:
    """Check if position `pos` is inside a quoted region."""
    # Count opening quotes before this position
    # Bangla uses: " " (smart quotes), ' ' (single smart), « »
    open_double = 0
    open_single = 0
    for i in range(pos):
        ch = text[i]
        if ch in ('"', '\u201c'):  # opening double
            open_double += 1
        elif ch in ('"', '\u201d'):  # closing double
            open_double -= 1
        elif ch in ("'", '\u2018'):  # opening single
            open_single += 1
        elif ch in ("'", '\u2019'):  # closing single
            open_single -= 1
    return open_double > 0 or open_single > 0


def _is_abbreviation(text: str, pos: int) -> bool:
    """Check if a period at `pos` is part of an abbreviation."""
    if pos < 1:
        return False
    # Look backwards to find the start of the word
    start = pos
    while start > 0 and text[start - 1] not in (' ', '\t', '\n'):
        start -= 1
    word = text[start:pos + 1]  # includes the period
    return word in BANGLA_ABBREVIATIONS


def split_sentences(text: str) -> List[str]:
    """
    Split Bangla text into sentences.
    
    Args:
        text: Raw Bangla text (may contain multiple paragraphs)
        
    Returns:
        List of sentence strings, stripped of leading/trailing whitespace.
        Empty sentences are filtered out.
    """
    if not text or not text.strip():
        return []

    # Normalize whitespace (but preserve newlines for paragraph splitting)
    text = text.strip()
    
    # Step 1: Split on paragraph boundaries (double newlines)
    paragraphs = re.split(r'\n\s*\n', text)
    
    all_sentences = []
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # Step 2: Split on single newlines within paragraph
        # (Bangla news articles often have one sentence per line)
        lines = paragraph.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Step 3: Split on sentence terminators
            sentences_from_line = _split_line(line)
            all_sentences.extend(sentences_from_line)
    
    return all_sentences


def _split_line(line: str) -> List[str]:
    """Split a single line into sentences on Bangla terminators."""
    sentences = []
    current = []
    i = 0
    
    while i < len(line):
        ch = line[i]
        current.append(ch)
        
        # Check if this character is a sentence terminator
        if ch in ('।', '॥', '?', '!'):
            # Don't split inside quotes
            if _is_inside_quotes(line, i):
                i += 1
                continue
            
            # This is a sentence boundary
            sentence = ''.join(current).strip()
            if sentence:
                sentences.append(sentence)
            current = []
        
        elif ch == '.':
            # Check: is this a decimal number? (e.g., 3.14)
            if i > 0 and i < len(line) - 1:
                if line[i - 1].isdigit() and line[i + 1].isdigit():
                    i += 1
                    continue
            # Check: is this an abbreviation?
            if _is_abbreviation(line, i):
                i += 1
                continue
            # English-style period as sentence ender in Bangla text
            # Only treat as boundary if followed by space + uppercase/Bangla char
            if i < len(line) - 1 and line[i + 1] in (' ', '\t'):
                sentence = ''.join(current).strip()
                if sentence:
                    sentences.append(sentence)
                current = []
        
        i += 1
    
    # Remaining text (sentence without terminator at end)
    remaining = ''.join(current).strip()
    if remaining:
        sentences.append(remaining)
    
    return sentences


# ============================================================================
# Convenience: tokenize with BanglaT5 tokenizer for accurate BPE token count
# ============================================================================

_tokenizer_cache = {}

def get_bangla_tokenizer():
    """Lazy-load and cache the BanglaT5 tokenizer."""
    if 'tok' not in _tokenizer_cache:
        from transformers import AutoTokenizer
        try:
            tok = AutoTokenizer.from_pretrained("csebuetnlp/banglaT5")
        except Exception:
            # Some environments attempt a slow->fast conversion that can require
            # optional deps (e.g., tiktoken). Fall back to the slow tokenizer.
            tok = AutoTokenizer.from_pretrained("csebuetnlp/banglaT5", use_fast=False)
        # Suppress "Token indices sequence length is longer than..." warning.
        # We only use the tokenizer for counting, not for model input.
        tok.model_max_length = 1_000_000
        _tokenizer_cache['tok'] = tok
    return _tokenizer_cache['tok']


def count_bpe_tokens(text: Union[str, List[str]], batch_size: int = 100) -> Union[int, List[int]]:
    """Count actual BPE tokens using the BanglaT5 tokenizer.
    
    Args:
        text: A single string or a list of strings.
        batch_size: For list input, process in batches to avoid OOM (default 100)
        
    Returns:
        Token count (int) for a single string, or list of counts for a batch.
    """
    tok = get_bangla_tokenizer()
    if isinstance(text, list):
        # Process large lists in batches to avoid tokenizer OOM
        if len(text) > batch_size:
            all_counts = []
            for i in range(0, len(text), batch_size):
                batch = text[i:i + batch_size]
                encoded = tok(batch, add_special_tokens=False)['input_ids']
                all_counts.extend([len(ids) for ids in encoded])
            return all_counts
        # Small batch — process directly
        encoded = tok(text, add_special_tokens=False)['input_ids']
        return [len(ids) for ids in encoded]
    return len(tok.encode(text, add_special_tokens=False))


# ============================================================================
# Quick self-test
# ============================================================================
if __name__ == "__main__":
    test_text = (
        "বাংলাদেশ দক্ষিণ এশিয়ার একটি দেশ। এটি ভারতের পূর্বে অবস্থিত। "
        "দেশটির রাজধানী ঢাকা। জনসংখ্যা প্রায় ১৭ কোটি।\n\n"
        "বাংলাদেশের আয়তন ১,৪৭,৫৭০ বর্গ কি.মি.। "
        "ড. মুহাম্মদ ইউনূস নোবেল পুরস্কার পেয়েছেন। "
        "তিনি কি আসবেন? হ্যাঁ, আসবেন!"
    )
    
    sentences = split_sentences(test_text)
    print(f"Found {len(sentences)} sentences:\n")
    for i, s in enumerate(sentences, 1):
        print(f"  [{i}] {s}")
    
    print(f"\nTotal BPE tokens: {count_bpe_tokens(test_text)}")
