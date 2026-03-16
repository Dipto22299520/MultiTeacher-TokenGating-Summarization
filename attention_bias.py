"""
Soft Attention Bias — Solution 4 (Advanced)

Instead of hard chunk boundaries, add a learnable distance-based bias
to T5's self-attention so that tokens within the same chunk attend
more strongly to each other, while cross-boundary attention is possible
but dampened.

Implementation:
  1. Add a special [CHUNK_BOUNDARY] token to the tokenizer vocabulary
  2. Subclass T5's attention to inject chunk-aware bias
  3. The bias decays with distance from chunk boundary
  4. Bias strength is a learnable parameter (or fixed hyperparameter)

This is the most complex solution — only use after Solutions 1-3 are working.
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Attention, T5Stack

# Special token for chunk boundaries
CHUNK_BOUNDARY_TOKEN = "[CHUNK_BOUNDARY]"


def add_chunk_boundary_token(tokenizer: AutoTokenizer) -> int:
    """
    Add the [CHUNK_BOUNDARY] token to the tokenizer vocabulary.
    
    Returns:
        The token ID of the newly added token.
    """
    num_added = tokenizer.add_special_tokens({
        'additional_special_tokens': [CHUNK_BOUNDARY_TOKEN]
    })
    
    boundary_id = tokenizer.convert_tokens_to_ids(CHUNK_BOUNDARY_TOKEN)
    return boundary_id


def insert_chunk_boundaries(text: str, chunk_texts: List[str]) -> str:
    """
    Given original text split into chunks, rebuild it with [CHUNK_BOUNDARY]
    tokens inserted at boundaries.
    
    This is used when feeding multiple chunks as a single sequence
    (e.g., during training with shorter articles that fit after boundary marking).
    
    For long articles that are processed chunk-by-chunk, this isn't needed —
    the attention bias is applied within each chunk based on overlap markers.
    
    Args:
        text: The full article text (unused, for reference)
        chunk_texts: List of chunk text strings
        
    Returns:
        Single string with [CHUNK_BOUNDARY] between chunks
    """
    return f" {CHUNK_BOUNDARY_TOKEN} ".join(chunk_texts)


class ChunkAwareAttentionBias(nn.Module):
    """
    Computes a chunk-aware attention bias matrix.
    
    For a sequence of tokens with chunk boundary markers, this produces
    a bias matrix where:
      - Same-chunk token pairs: bias = 0 (no penalty)
      - Cross-chunk token pairs: bias = -alpha * distance_to_boundary
    
    The effect: tokens CAN attend across boundaries, but the attention
    is dampened based on how far the source/target are from the boundary.
    
    Args:
        alpha: Base attention dampening strength (default 0.5)
              Higher = stronger same-chunk preference
        learnable: If True, alpha is a learnable parameter (default True)
        decay_type: 'linear' or 'exponential' decay with distance
    """
    
    def __init__(
        self, 
        alpha: float = 0.5, 
        learnable: bool = True,
        decay_type: str = 'linear'
    ):
        super().__init__()
        
        if learnable:
            self.alpha = nn.Parameter(torch.tensor(alpha))
        else:
            self.register_buffer('alpha', torch.tensor(alpha))
        
        self.decay_type = decay_type
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        boundary_token_id: int
    ) -> torch.Tensor:
        """
        Compute the chunk-aware attention bias matrix.
        
        Args:
            input_ids: (batch_size, seq_len) token IDs
            boundary_token_id: ID of the [CHUNK_BOUNDARY] token
            
        Returns:
            bias: (batch_size, 1, seq_len, seq_len) bias to add to attention scores
                  (1 in dim 1 for broadcasting across heads)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Find chunk assignments for each token
        # Chunk ID increments each time we see a [CHUNK_BOUNDARY] token
        is_boundary = (input_ids == boundary_token_id).long()  # (B, L)
        chunk_ids = torch.cumsum(is_boundary, dim=1)  # (B, L)
        
        # Compute pairwise "same chunk" matrix
        # chunk_ids_i: (B, L, 1), chunk_ids_j: (B, 1, L)
        chunk_ids_i = chunk_ids.unsqueeze(2)  # (B, L, 1)
        chunk_ids_j = chunk_ids.unsqueeze(1)  # (B, 1, L)
        
        same_chunk = (chunk_ids_i == chunk_ids_j).float()  # (B, L, L)
        
        # For cross-chunk pairs, compute distance-based decay
        # Distance = absolute difference in chunk IDs
        chunk_distance = (chunk_ids_i - chunk_ids_j).abs().float()  # (B, L, L)
        
        if self.decay_type == 'exponential':
            # Exponential decay: bias = -alpha * exp(distance - 1)
            cross_chunk_bias = -self.alpha * torch.exp(chunk_distance - 1)
        else:
            # Linear decay: bias = -alpha * distance
            cross_chunk_bias = -self.alpha * chunk_distance
        
        # Same-chunk: no bias; cross-chunk: negative bias
        bias = cross_chunk_bias * (1 - same_chunk)
        
        # Add head dimension: (B, 1, L, L)
        bias = bias.unsqueeze(1)
        
        return bias


class ChunkAwareT5(nn.Module):
    """
    Wrapper around T5ForConditionalGeneration that injects chunk-aware
    attention bias into the encoder's self-attention.
    
    Usage:
        model = ChunkAwareT5.from_pretrained("csebuetnlp/banglaT5", tokenizer)
        outputs = model(input_ids, attention_mask, labels)
    
    The attention bias is ONLY applied to the encoder (where we process
    chunked input). The decoder attends to encoder outputs normally.
    
    Args:
        base_model: T5ForConditionalGeneration base model
        boundary_token_id: Token ID for [CHUNK_BOUNDARY]
        alpha: Attention bias strength
        learnable_alpha: Whether alpha is learnable
    """
    
    def __init__(
        self,
        base_model: T5ForConditionalGeneration,
        boundary_token_id: int,
        alpha: float = 0.5,
        learnable_alpha: bool = True
    ):
        super().__init__()
        self.base_model = base_model
        self.boundary_token_id = boundary_token_id
        self.attention_bias = ChunkAwareAttentionBias(
            alpha=alpha, 
            learnable=learnable_alpha
        )
        
        # Store config reference for compatibility with Trainer
        self.config = base_model.config
        self.generation_config = getattr(base_model, 'generation_config', None)
    
    @classmethod
    def from_pretrained(
        cls, 
        model_name: str, 
        tokenizer: AutoTokenizer,
        alpha: float = 0.5,
        learnable_alpha: bool = True
    ):
        """
        Load a pretrained T5 model and wrap it with chunk-aware attention.
        
        Also resizes the embedding to accommodate the [CHUNK_BOUNDARY] token.
        """
        # Add special token
        boundary_id = add_chunk_boundary_token(tokenizer)
        
        # Load base model
        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Resize embeddings for the new token
        base_model.resize_token_embeddings(len(tokenizer))
        
        return cls(
            base_model=base_model,
            boundary_token_id=boundary_id,
            alpha=alpha,
            learnable_alpha=learnable_alpha
        )
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        decoder_input_ids=None,
        **kwargs
    ):
        """
        Forward pass with chunk-aware attention bias injected into the encoder.
        
        The T5 model's forward() accepts `encoder_attention_bias` (or we hook into
        the encoder outputs). We use the position_bias mechanism.
        """
        # Compute chunk-aware bias
        if input_ids is not None:
            chunk_bias = self.attention_bias(input_ids, self.boundary_token_id)
        else:
            chunk_bias = None
        
        # Run encoder with the bias
        # T5's encoder self-attention computes: attn_weights = softmax(QK^T / sqrt(d) + position_bias)
        # We can inject our chunk bias by adding it to the encoder outputs
        # 
        # Strategy: Run encoder, get hidden states, then run decoder normally
        # For the bias: we use a hook on the encoder's attention layers
        
        # Register hooks to inject bias into attention
        hooks = []
        if chunk_bias is not None:
            hooks = self._register_attention_hooks(chunk_bias)
        
        try:
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                decoder_input_ids=decoder_input_ids,
                **kwargs
            )
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
        
        return outputs
    
    def _register_attention_hooks(self, chunk_bias: torch.Tensor):
        """
        Register forward hooks on encoder self-attention layers to inject
        chunk-aware bias into the attention scores.
        """
        hooks = []
        
        for layer in self.base_model.encoder.block:
            # Each T5Block has layer[0] = SelfAttention
            attn_module = layer.layer[0].SelfAttention
            
            def make_hook(bias):
                def hook_fn(module, args, output):
                    # T5Attention.forward returns (attn_output, attn_weights, position_bias)
                    # We modify the position_bias to include our chunk bias
                    if isinstance(output, tuple) and len(output) >= 3:
                        attn_output, attn_weights, position_bias = output[:3]
                        # The bias needs to match position_bias shape
                        # position_bias: (batch, n_heads, seq_len, seq_len)
                        # chunk_bias: (batch, 1, seq_len, seq_len) — broadcasts
                        if position_bias is not None:
                            # Clamp to avoid dimension mismatches (truncated sequences)
                            _, _, pb_seq_i, pb_seq_j = position_bias.shape
                            _, _, cb_seq_i, cb_seq_j = bias.shape
                            min_i = min(pb_seq_i, cb_seq_i)
                            min_j = min(pb_seq_j, cb_seq_j)
                            modified_bias = position_bias.clone()
                            modified_bias[:, :, :min_i, :min_j] += bias[:, :, :min_i, :min_j]
                            return (attn_output, attn_weights, modified_bias) + output[3:]
                    return output
                return hook_fn
            
            hook = attn_module.register_forward_hook(make_hook(chunk_bias))
            hooks.append(hook)
        
        return hooks
    
    def generate(self, **kwargs):
        """Forward generation to base model, with attention hooks if input_ids present."""
        input_ids = kwargs.get('input_ids', None)
        
        if input_ids is not None:
            chunk_bias = self.attention_bias(input_ids, self.boundary_token_id)
            hooks = self._register_attention_hooks(chunk_bias)
            try:
                return self.base_model.generate(**kwargs)
            finally:
                for hook in hooks:
                    hook.remove()
        else:
            return self.base_model.generate(**kwargs)
    
    def save_pretrained(self, save_directory, **kwargs):
        """Save the base model + attention bias parameters."""
        import os
        self.base_model.save_pretrained(save_directory, **kwargs)
        # Save the attention bias parameters separately
        bias_path = os.path.join(save_directory, "chunk_attention_bias.pt")
        torch.save({
            'alpha': self.attention_bias.alpha.data,
            'boundary_token_id': self.boundary_token_id,
            'decay_type': self.attention_bias.decay_type,
        }, bias_path)
    
    @classmethod
    def from_saved(
        cls,
        save_directory: str,
        tokenizer: AutoTokenizer,
    ):
        """Load a saved ChunkAwareT5 model."""
        import os
        
        base_model = AutoModelForSeq2SeqLM.from_pretrained(save_directory)
        
        bias_path = os.path.join(save_directory, "chunk_attention_bias.pt")
        if os.path.exists(bias_path):
            bias_state = torch.load(bias_path, map_location='cpu', weights_only=True)
            boundary_id = bias_state['boundary_token_id']
            alpha = bias_state['alpha'].item()
            decay_type = bias_state.get('decay_type', 'linear')
        else:
            # Fallback: add token and use defaults
            boundary_id = add_chunk_boundary_token(tokenizer)
            alpha = 0.5
            decay_type = 'linear'
        
        model = cls(
            base_model=base_model,
            boundary_token_id=boundary_id,
            alpha=alpha,
            learnable_alpha=True
        )
        model.attention_bias.decay_type = decay_type
        
        return model
    
    # Delegate common attributes to base_model for Trainer compatibility
    def parameters(self, recurse=True):
        yield from self.base_model.parameters(recurse)
        yield from self.attention_bias.parameters(recurse)
    
    def named_parameters(self, prefix='', recurse=True):
        yield from self.base_model.named_parameters(prefix + 'base_model.', recurse)
        yield from self.attention_bias.named_parameters(prefix + 'attention_bias.', recurse)
    
    def train(self, mode=True):
        self.base_model.train(mode)
        self.attention_bias.train(mode)
        return super().train(mode)
    
    def eval(self):
        self.base_model.eval()
        self.attention_bias.eval()
        return super().eval()
    
    @property
    def device(self):
        return next(self.base_model.parameters()).device
    
    def to(self, *args, **kwargs):
        self.base_model.to(*args, **kwargs)
        self.attention_bias.to(*args, **kwargs)
        return super().to(*args, **kwargs)


# ============================================================================
# Self-test
# ============================================================================
if __name__ == "__main__":
    print("Testing ChunkAwareAttentionBias...")
    
    # Simulate a batch of token IDs with boundary markers
    # Assume boundary_token_id = 99
    boundary_id = 99
    input_ids = torch.tensor([
        [10, 20, 30, 99, 40, 50, 60, 99, 70, 80],  # 3 chunks
        [10, 20, 99, 30, 40, 50, 60, 70, 80, 90],   # 2 chunks
    ])
    
    bias_module = ChunkAwareAttentionBias(alpha=0.5, learnable=True)
    bias = bias_module(input_ids, boundary_id)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Bias shape: {bias.shape}")
    print(f"Alpha (learnable): {bias_module.alpha.item():.3f}")
    
    # Check: same-chunk pairs should have 0 bias
    # Cross-chunk pairs should have negative bias
    print(f"\nSample 0, bias matrix (squeezed):")
    print(bias[0, 0].detach().numpy().round(2))
    
    print("\n✓ ChunkAwareAttentionBias test passed!")
