"""
Unified Evaluation Module for KV Cache Compression

This module provides evaluation functions that work with any compression method.
It measures PPL (perplexity) and accuracy with KV cache compression.

Key implementation points:
1. Process one token at a time (autoregressive)
2. Use CrossEntropyLoss(reduction="none") for per-token NLL
3. Apply compression after EVERY forward pass (if enabled)
4. PPL = exp(mean(nlls))
5. Accuracy = num_correct / num_tokens
"""

from typing import Callable, Dict, List, Optional, Union
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import DynamicCache

from .utils import to_dynamic_cache, normalize_kv_cache


def evaluate_with_compression(
    model,
    tokenizer,
    text: str,
    compress_fn: Optional[Callable] = None,
    compress_kwargs: Optional[Dict] = None,
    max_tokens: int = 3000,
    skip_layers: List[int] = [0, 1],
    device: Optional[torch.device] = None,
    show_progress: bool = True,
) -> Dict[str, float]:
    """
    Evaluate PPL and Accuracy with KV cache compression.
    
    This function supports any compression method through the compress_fn parameter.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        text: Input text to evaluate
        compress_fn: Compression function with signature:
                    fn(past_key_values, skip_layers=..., **kwargs) -> compressed_kv
                    If None, no compression is applied (baseline evaluation)
        compress_kwargs: Additional kwargs to pass to compress_fn
                        (e.g., keep_ratio, fix_kv_size, start_size, etc.)
        max_tokens: Maximum number of tokens to evaluate
        skip_layers: Layer indices to skip compression
        device: Device to use (auto-detected if None)
        show_progress: Whether to show progress bar
    
    Returns:
        Dict containing:
        - perplexity: The perplexity score
        - accuracy: Next token prediction accuracy
        - num_tokens: Number of tokens evaluated
        - final_cache_size: Final KV cache size after compression
    
    Example:
        >>> from kvcompress.methods import l2_compress
        >>> results = evaluate_with_compression(
        ...     model, tokenizer, text,
        ...     compress_fn=l2_compress,
        ...     compress_kwargs={"keep_ratio": 0.8, "prune_after": 1000}
        ... )
        
        >>> from kvcompress.methods import streaming_llm_compress
        >>> results = evaluate_with_compression(
        ...     model, tokenizer, text,
        ...     compress_fn=streaming_llm_compress,
        ...     compress_kwargs={"start_size": 4, "recent_size": 508}
        ... )
    """
    if device is None:
        device = next(model.parameters()).device
    
    if compress_kwargs is None:
        compress_kwargs = {}
    
    # Tokenize input
    input_ids = tokenizer.encode(text, return_tensors="pt")
    input_ids = input_ids[:, :max_tokens].to(device)
    seq_len = input_ids.shape[1]
    
    if seq_len < 2:
        return {
            "perplexity": float('inf'),
            "accuracy": 0.0,
            "num_tokens": 0,
            "final_cache_size": 0
        }
    
    # Loss function for per-token NLL
    loss_fn = CrossEntropyLoss(reduction="none")
    
    # Initialize
    past_key_values = None
    nlls = []
    num_correct = []
    
    model.eval()
    
    # Create progress bar if requested
    token_range = range(seq_len - 1)
    if show_progress:
        token_range = tqdm(token_range, desc="Evaluating")
    
    with torch.inference_mode():
        for idx in token_range:
            # Current token (single token input)
            current_token = input_ids[:, idx:idx+1]
            
            # Target token (next token)
            target = input_ids[:, idx+1:idx+2].view(-1)
            
            # Forward pass
            outputs = model(
                current_token,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            # Get logits for next token prediction
            logits = outputs.logits[:, -1, :].view(-1, model.config.vocab_size)
            
            # Calculate NLL for this token
            nll = loss_fn(logits, target)
            nlls.append(nll.item())
            
            # Calculate accuracy
            predicted = torch.argmax(logits, dim=-1)
            num_correct.append((predicted == target).int().item())
            
            # Update progress bar
            if show_progress:
                current_ppl = torch.exp(torch.tensor(nlls).mean()).item()
                current_acc = sum(num_correct) / len(num_correct)
                token_range.set_description(
                    f"PPL: {current_ppl:.2f}, Acc: {current_acc:.2%}"
                )
            
            # Get KV cache from outputs
            past_key_values = outputs.past_key_values
            
            # Apply compression if compress_fn is provided
            if compress_fn is not None and past_key_values is not None:
                # Convert to list format for compression
                kv_list = list(normalize_kv_cache(past_key_values))
                
                # Apply compression
                compressed_kv = compress_fn(
                    kv_list,
                    skip_layers=skip_layers,
                    **compress_kwargs
                )
                
                # Convert back to DynamicCache for next iteration
                past_key_values = to_dynamic_cache(compressed_kv)
    
    # Calculate final metrics
    perplexity = torch.exp(torch.tensor(nlls).mean()).item()
    accuracy = sum(num_correct) / len(num_correct)
    
    # Get final cache size
    final_cache_size = 0
    if past_key_values is not None:
        kv_list = list(normalize_kv_cache(past_key_values))
        if kv_list:
            final_cache_size = kv_list[0][0].size(2)
    
    return {
        "perplexity": perplexity,
        "accuracy": accuracy,
        "num_tokens": len(nlls),
        "final_cache_size": final_cache_size
    }


def evaluate_baseline(
    model,
    tokenizer,
    text: str,
    max_tokens: int = 3000,
    device: Optional[torch.device] = None,
    show_progress: bool = False,
) -> Dict[str, float]:
    """
    Evaluate baseline PPL and Accuracy without compression.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        text: Input text to evaluate
        max_tokens: Maximum number of tokens to evaluate
        device: Device to use
        show_progress: Whether to show progress bar
    
    Returns:
        Dict with perplexity, accuracy, num_tokens, final_cache_size
    """
    return evaluate_with_compression(
        model=model,
        tokenizer=tokenizer,
        text=text,
        compress_fn=None,  # No compression
        max_tokens=max_tokens,
        device=device,
        show_progress=show_progress,
    )


def compare_methods(
    model,
    tokenizer,
    text: str,
    methods_config: List[Dict],
    max_tokens: int = 3000,
    skip_layers: List[int] = [0, 1],
    device: Optional[torch.device] = None,
) -> List[Dict[str, float]]:
    """
    Compare PPL and Accuracy across different compression methods.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        text: Input text to evaluate
        methods_config: List of dicts, each containing:
                       - "name": Method name for display
                       - "compress_fn": Compression function
                       - "kwargs": Dict of compression parameters
        max_tokens: Maximum number of tokens to evaluate
        skip_layers: Layer indices to skip compression
        device: Device to use
    
    Returns:
        List of result dicts, one per method
    
    Example:
        >>> from kvcompress.methods import l2_compress, streaming_llm_compress
        >>> methods = [
        ...     {"name": "baseline", "compress_fn": None, "kwargs": {}},
        ...     {"name": "l2_0.8", "compress_fn": l2_compress, "kwargs": {"keep_ratio": 0.8}},
        ...     {"name": "streaming", "compress_fn": streaming_llm_compress, "kwargs": {"start_size": 4, "recent_size": 508}},
        ... ]
        >>> results = compare_methods(model, tokenizer, text, methods)
    """
    results = []
    
    for config in methods_config:
        name = config.get("name", "unknown")
        compress_fn = config.get("compress_fn", None)
        kwargs = config.get("kwargs", {})
        
        print(f"\nEvaluating {name}...")
        
        result = evaluate_with_compression(
            model=model,
            tokenizer=tokenizer,
            text=text,
            compress_fn=compress_fn,
            compress_kwargs=kwargs,
            max_tokens=max_tokens,
            skip_layers=skip_layers,
            device=device,
            show_progress=True,
        )
        
        result['method'] = name
        result['config'] = kwargs
        results.append(result)
        
        print(f"  PPL: {result['perplexity']:.2f}")
        print(f"  Accuracy: {result['accuracy']:.2%}")
        print(f"  Final cache size: {result['final_cache_size']}")
    
    return results


__all__ = [
    'evaluate_with_compression',
    'evaluate_baseline',
    'compare_methods',
]

