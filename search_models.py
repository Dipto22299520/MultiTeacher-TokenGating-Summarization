"""Search for alternative small models for summarization"""

from huggingface_hub import HfApi
import sys

def search_models(query, limit=30):
    """Search for models on Hugging Face Hub"""
    api = HfApi()
    
    print(f"\nSearching for: {query}")
    print("="*80)
    
    try:
        models = list(api.list_models(search=query, limit=limit, sort="downloads", direction=-1))
        
        if not models:
            print(f"No models found for '{query}'")
            return
        
        print(f"\nFound {len(models)} models:\n")
        for i, model in enumerate(models[:20], 1):
            print(f"{i}. {model.id}")
            if hasattr(model, 'downloads') and model.downloads:
                print(f"   Downloads: {model.downloads:,}")
            if hasattr(model, 'tags') and model.tags:
                tags = [t for t in model.tags if t in ['text2text-generation', 'summarization', 'seq2seq', 'translation']]
                if tags:
                    print(f"   Tags: {', '.join(tags)}")
            print()
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    queries = ["gemma", "gemma-2b", "mbart", "small summarization"]
    
    if len(sys.argv) > 1:
        queries = [sys.argv[1]]
    
    for query in queries:
        search_models(query)
        print("\n" + "="*80 + "\n")
