"""
Comprehensive evaluation of BanglaT5 model on full articles (entire context).

For articles >1000 tokens: Uses MapReduce inference pipeline (all chunks → final summary)
For articles ≤1000 tokens: Direct summarization

Metrics:
1. BLEU (n-gram overlap)
2. ROUGE (recall-oriented n-gram overlap)
3. BERTScore (contextual embedding similarity)
4. BARTScore (learned semantic evaluation)
5. Semantic Similarity (sentence embeddings cosine similarity)
"""

import os
import json
import unicodedata
import numpy as np
from tqdm import tqdm
from typing import List, Dict
import torch

# BLEU
try:
    from sacrebleu import corpus_bleu, sentence_bleu
    BLEU_AVAILABLE = True
except ImportError:
    print("⚠️  sacrebleu not installed. Install: pip install sacrebleu")
    BLEU_AVAILABLE = False

# ROUGE
from rouge_score import rouge_scorer

# BERTScore
try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    print("⚠️  bert_score not installed. Install: pip install bert-score")
    BERTSCORE_AVAILABLE = False

# BARTScore
try:
    from bart_score import BARTScorer
    BARTSCORE_AVAILABLE = True
except ImportError:
    print("⚠️  bart_score not installed. Install: pip install bart-score")
    BARTSCORE_AVAILABLE = False

# Semantic Similarity (Sentence Transformers)
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SEMANTIC_AVAILABLE = True
except ImportError:
    print("⚠️  sentence-transformers not installed. Install: pip install sentence-transformers")
    SEMANTIC_AVAILABLE = False

# Our inference pipeline
from inference_pipeline import HierarchicalSummarizer
from bangla_sentence_splitter import BanglaSentenceSplitter


def normalize_bangla(text):
    """Normalize Bangla text."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace('\u200d', '').replace('\u200c', '')
    text = ' '.join(text.split())
    return text.strip()


class CharTokenizer:
    """Character-level tokenizer for Bangla ROUGE."""
    def tokenize(self, text):
        return list(text)


class FullArticleEvaluator:
    """
    Evaluate model on full articles with complete context.
    """
    
    def __init__(
        self,
        model_path: str,
        test_data_path: str,
        output_path: str = "full_article_evaluation_results.json",
        max_samples: int = None
    ):
        """
        Args:
            model_path: Path to trained model checkpoint
            test_data_path: Path to test data JSON/JSONL (original articles, not chunks)
            output_path: Where to save evaluation results
            max_samples: Limit evaluation to N samples (for debugging)
        """
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.output_path = output_path
        self.max_samples = max_samples
        
        print("\n" + "=" * 80)
        print("FULL-ARTICLE EVALUATION SETUP")
        print("=" * 80)
        print(f"Model: {model_path}")
        print(f"Test data: {test_data_path}")
        print(f"Max samples: {max_samples if max_samples else 'All'}")
        
        # Initialize inference pipeline
        print("\n Loading hierarchical summarizer...")
        self.summarizer = HierarchicalSummarizer(model_path)
        
        # Initialize metrics
        print("\nInitializing metrics...")
        
        # ROUGE (character-level for Bangla)
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=False,
            tokenizer=CharTokenizer()
        )
        print("  ✓ ROUGE (character-level)")
        
        # BERTScore
        if BERTSCORE_AVAILABLE:
            print("  ✓ BERTScore")
        else:
            print("  ✗ BERTScore (not available)")
        
        # BARTScore
        self.bart_scorer = None
        if BARTSCORE_AVAILABLE:
            try:
                print("  Loading BARTScore model...")
                # Use multilingual BART for Bangla
                self.bart_scorer = BARTScorer(device='cuda' if torch.cuda.is_available() else 'cpu', 
                                             checkpoint='facebook/mbart-large-cc25')
                print("  ✓ BARTScore")
            except Exception as e:
                print(f"  ✗ BARTScore (error: {e})")
        else:
            print("  ✗ BARTScore (not available)")
        
        # Semantic Similarity
        self.semantic_model = None
        if SEMANTIC_AVAILABLE:
            try:
                print("  Loading semantic similarity model...")
                # Use multilingual sentence transformer
                self.semantic_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
                print("  ✓ Semantic Similarity")
            except Exception as e:
                print(f"  ✗ Semantic Similarity (error: {e})")
        else:
            print("  ✗ Semantic Similarity (not available)")
        
        # BLEU
        if BLEU_AVAILABLE:
            print("  ✓ BLEU")
        else:
            print("  ✗ BLEU (not available)")
    
    def load_test_data(self) -> List[Dict]:
        """Load test data (original articles, not chunked)."""
        print("\n" + "=" * 80)
        print("LOADING TEST DATA")
        print("=" * 80)
        
        # Detect format
        if self.test_data_path.endswith('.jsonl'):
            print("Format: JSONL (one article per line)")
            data = []
            with open(self.test_data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        else:
            print("Format: JSON (array)")
            with open(self.test_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        print(f"Total articles: {len(data)}")
        
        # Apply sample limit
        if self.max_samples:
            data = data[:self.max_samples]
            print(f"Limited to: {len(data)} articles")
        
        # Validate data
        valid_data = []
        for item in data:
            if 'text' in item and 'summary' in item:
                if item['text'] and item['summary']:
                    valid_data.append(item)
        
        print(f"Valid articles: {len(valid_data)}")
        return valid_data
    
    def generate_summary(self, article_text: str, article_id: int = None) -> str:
        """
        Generate summary for full article using inference pipeline.
        Automatically handles long articles with MapReduce.
        """
        try:
            summary = self.summarizer.summarize(article_text)
            return normalize_bangla(summary)
        except Exception as e:
            print(f"\n⚠️  Error summarizing article {article_id}: {e}")
            return ""
    
    def compute_bleu(self, predictions: List[str], references: List[str]) -> Dict:
        """Compute BLEU scores."""
        if not BLEU_AVAILABLE:
            return {}
        
        # sacrebleu expects references as list of lists
        refs = [[ref] for ref in references]
        
        # Corpus BLEU
        bleu_result = corpus_bleu(predictions, refs)
        
        # Sentence-level BLEU (average)
        sentence_bleus = []
        for pred, ref in zip(predictions, references):
            try:
                sent_bleu = sentence_bleu(pred, [ref])
                sentence_bleus.append(sent_bleu.score)
            except:
                sentence_bleus.append(0.0)
        
        return {
            'bleu_corpus': bleu_result.score,
            'bleu_avg': np.mean(sentence_bleus)
        }
    
    def compute_rouge(self, predictions: List[str], references: List[str]) -> Dict:
        """Compute ROUGE scores (character-level for Bangla)."""
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        return {
            'rouge1': np.mean(rouge1_scores),
            'rouge2': np.mean(rouge2_scores),
            'rougeL': np.mean(rougeL_scores)
        }
    
    def compute_bertscore(self, predictions: List[str], references: List[str]) -> Dict:
        """Compute BERTScore."""
        if not BERTSCORE_AVAILABLE:
            return {}
        
        print("  Computing BERTScore (this may take a while)...")
        P, R, F1 = bert_score(
            predictions,
            references,
            lang='en',  # Use multilingual BERT
            verbose=False,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        return {
            'bertscore_precision': P.mean().item(),
            'bertscore_recall': R.mean().item(),
            'bertscore_f1': F1.mean().item()
        }
    
    def compute_bartscore(self, predictions: List[str], references: List[str]) -> Dict:
        """Compute BARTScore."""
        if not self.bart_scorer:
            return {}
        
        print("  Computing BARTScore (this may take a while)...")
        try:
            # BARTScore: ref→pred direction (measures faithfulness)
            scores = self.bart_scorer.score(references, predictions, batch_size=4)
            return {
                'bartscore': np.mean(scores)
            }
        except Exception as e:
            print(f"  ⚠️  BARTScore error: {e}")
            return {}
    
    def compute_semantic_similarity(self, predictions: List[str], references: List[str]) -> Dict:
        """Compute semantic similarity using sentence embeddings."""
        if not self.semantic_model:
            return {}
        
        print("  Computing semantic similarity...")
        try:
            # Encode in batches
            pred_embeddings = self.semantic_model.encode(predictions, batch_size=32, show_progress_bar=False)
            ref_embeddings = self.semantic_model.encode(references, batch_size=32, show_progress_bar=False)
            
            # Compute cosine similarity
            similarities = []
            for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
                sim = cosine_similarity([pred_emb], [ref_emb])[0][0]
                similarities.append(sim)
            
            return {
                'semantic_similarity': np.mean(similarities),
                'semantic_similarity_std': np.std(similarities)
            }
        except Exception as e:
            print(f"  ⚠️  Semantic similarity error: {e}")
            return {}
    
    def evaluate(self):
        """Run full evaluation."""
        print("\n" + "=" * 80)
        print("STARTING EVALUATION")
        print("=" * 80)
        
        # Load test data
        test_data = self.load_test_data()
        
        if len(test_data) == 0:
            print("❌ No valid test data found!")
            return
        
        # Generate summaries
        print("\n" + "=" * 80)
        print("GENERATING SUMMARIES")
        print("=" * 80)
        
        predictions = []
        references = []
        articles = []
        
        for i, item in enumerate(tqdm(test_data, desc="Summarizing articles")):
            article_text = normalize_bangla(str(item['text']))
            reference_summary = normalize_bangla(str(item['summary']))
            
            # Generate summary for full article
            predicted_summary = self.generate_summary(article_text, article_id=i)
            
            predictions.append(predicted_summary)
            references.append(reference_summary)
            articles.append(article_text)
        
        # Compute metrics
        print("\n" + "=" * 80)
        print("COMPUTING METRICS")
        print("=" * 80)
        
        results = {}
        
        # ROUGE
        print("\n1. Computing ROUGE...")
        rouge_results = self.compute_rouge(predictions, references)
        results.update(rouge_results)
        for key, value in rouge_results.items():
            print(f"  {key}: {value:.4f}")
        
        # BLEU
        if BLEU_AVAILABLE:
            print("\n2. Computing BLEU...")
            bleu_results = self.compute_bleu(predictions, references)
            results.update(bleu_results)
            for key, value in bleu_results.items():
                print(f"  {key}: {value:.4f}")
        
        # BERTScore
        if BERTSCORE_AVAILABLE:
            print("\n3. Computing BERTScore...")
            bertscore_results = self.compute_bertscore(predictions, references)
            results.update(bertscore_results)
            for key, value in bertscore_results.items():
                print(f"  {key}: {value:.4f}")
        
        # BARTScore
        if self.bart_scorer:
            print("\n4. Computing BARTScore...")
            bartscore_results = self.compute_bartscore(predictions, references)
            results.update(bartscore_results)
            for key, value in bartscore_results.items():
                print(f"  {key}: {value:.4f}")
        
        # Semantic Similarity
        if self.semantic_model:
            print("\n5. Computing Semantic Similarity...")
            semantic_results = self.compute_semantic_similarity(predictions, references)
            results.update(semantic_results)
            for key, value in semantic_results.items():
                print(f"  {key}: {value:.4f}")
        
        # Summary statistics
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        
        pred_lengths = [len(p.split()) for p in predictions]
        ref_lengths = [len(r.split()) for r in references]
        
        stats = {
            'num_samples': len(predictions),
            'avg_pred_length': np.mean(pred_lengths),
            'avg_ref_length': np.mean(ref_lengths),
            'length_ratio': np.mean(pred_lengths) / np.mean(ref_lengths),
            'empty_predictions': sum(1 for p in predictions if len(p.strip()) == 0)
        }
        results.update(stats)
        
        print(f"  Samples evaluated: {stats['num_samples']}")
        print(f"  Avg predicted length: {stats['avg_pred_length']:.1f} words")
        print(f"  Avg reference length: {stats['avg_ref_length']:.1f} words")
        print(f"  Length ratio: {stats['length_ratio']:.3f}")
        print(f"  Empty predictions: {stats['empty_predictions']}")
        
        # Save results
        print("\n" + "=" * 80)
        print("SAVING RESULTS")
        print("=" * 80)
        
        output_data = {
            'metrics': results,
            'samples': [
                {
                    'article': art[:500] + '...' if len(art) > 500 else art,
                    'reference': ref,
                    'prediction': pred
                }
                for art, ref, pred in zip(articles[:10], references[:10], predictions[:10])
            ]  # Save first 10 samples for inspection
        }
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Results saved to: {self.output_path}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        for key, value in sorted(results.items()):
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        print("\n✅ EVALUATION COMPLETE!")
        return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate BanglaT5 on full articles")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint (e.g., ./banglaT5_chunked_*/final_model)"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Path to test data JSON/JSONL (original articles, NOT chunked data)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="full_article_evaluation_results.json",
        help="Output path for results"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit evaluation to N samples (for debugging)"
    )
    
    args = parser.parse_args()
    
    evaluator = FullArticleEvaluator(
        model_path=args.model,
        test_data_path=args.test_data,
        output_path=args.output,
        max_samples=args.max_samples
    )
    
    evaluator.evaluate()
