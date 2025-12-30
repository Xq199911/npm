"""
Real Model KV Extraction for MP-KVM Experiments

This module provides utilities to extract real KV vectors from Llama models
for use in compression experiments instead of synthetic data.
"""
from __future__ import annotations
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, Optional
import os


def setup_model_and_tokenizer(model_path: str, device: str = "cuda") -> Tuple:
    """
    Load Llama model and tokenizer.

    Args:
        model_path: Path to the model directory
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model from {model_path}...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with device mapping
    if device == "cuda" and torch.cuda.is_available():
        print(f"Loading model to CUDA device: {torch.cuda.get_device_name()}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # Use float16 for memory efficiency
            device_map={"": "cuda:0"},  # Force all on GPU
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map={"": device},
            trust_remote_code=True
        )

    print(f"Model loaded successfully. Vocab size: {len(tokenizer)}")
    return model, tokenizer


def create_long_context_text(length: int = 8000) -> str:
    """
    Create a long context text for KV extraction experiments.

    Args:
        length: Approximate number of tokens

    Returns:
        Long context text string
    """
    # Create a diverse long context with different topics
    topics = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models that computers use to perform specific tasks without explicit instructions. It relies on patterns and inference instead of explicit programming. Deep learning, a subset of machine learning, uses neural networks with multiple layers to model complex patterns in data.",
        "The history of computer science dates back to ancient times with the development of computing devices. The first programmable computer was the Jacquard loom from 1801, which used punched cards to control weaving patterns. Modern computer science emerged in the mid-20th century with the development of electronic computers and programming languages.",
        "Climate change refers to long-term shifts in temperatures and weather patterns, primarily caused by human activities such as burning fossil fuels. The Earth's average surface temperature has risen by about 1.1 degrees Celsius since the late 19th century, leading to more frequent extreme weather events, rising sea levels, and impacts on ecosystems worldwide.",
        "Quantum computing represents a revolutionary approach to computation that leverages quantum mechanics principles. Unlike classical bits that can be either 0 or 1, quantum bits (qubits) can exist in multiple states simultaneously through superposition, potentially solving certain computational problems much faster than classical computers.",
        "The field of bioinformatics combines biology, computer science, and information technology to understand biological data. It involves developing algorithms, databases, and tools to understand biological systems, including DNA sequencing analysis, protein structure prediction, and drug discovery through computational methods.",
        "Cryptocurrency represents a digital or virtual form of currency that uses cryptography for security. Bitcoin, the first cryptocurrency, was created in 2009 by an anonymous person or group using the pseudonym Satoshi Nakamoto. Blockchain technology underlies most cryptocurrencies, providing decentralized and transparent transaction records.",
        "Neuroscience studies the nervous system and brain function. It encompasses multiple disciplines including neurology, psychology, and biology. Recent advances in neuroimaging techniques like fMRI and EEG have provided unprecedented insights into brain activity and cognitive processes.",
        "Sustainable energy sources include solar, wind, hydro, geothermal, and biomass power. The transition to renewable energy is driven by concerns about climate change, energy security, and economic benefits. Solar photovoltaic systems convert sunlight directly into electricity, while wind turbines harness kinetic energy from moving air."
    ]

    # Repeat and combine topics to create long context
    context_parts = []
    while len(" ".join(context_parts)) < length * 4:  # Rough character estimate
        for topic in topics:
            context_parts.append(topic)
            if len(" ".join(context_parts)) >= length * 4:
                break

    full_text = " ".join(context_parts)
    return full_text[:length * 6]  # Ensure sufficient length


class RealModelKVExtractor:
    """
    Extract KV vectors from real Llama model attention layers.
    """

    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Set model to evaluation mode
        self.model.eval()

        # Get model configuration
        self.config = model.config
        self.num_layers = self.config.num_hidden_layers
        self.hidden_size = self.config.hidden_size

        # Ensure model is on the correct device
        if hasattr(self.model, 'hf_device_map'):
            print(f"Model device map: {self.model.hf_device_map}")
        else:
            print(f"Model device: {next(self.model.parameters()).device}")

        print(f"KV Extractor initialized: {self.num_layers} layers, hidden_size={self.hidden_size}")

    def extract_kv_from_text(self, text: str, max_length: int = 2048, layer_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract KV vectors from model attention layers for given text.

        Args:
            text: Input text
            max_length: Maximum sequence length to process
            layer_idx: Which layer to extract from (0 for first layer)

        Returns:
            Tuple of (keys, values) as numpy arrays
        """
        print(f"Extracting KV vectors from layer {layer_idx}, max_length={max_length}")

        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=False
        )

        # Get the actual device where model parameters are located
        model_device = next(self.model.parameters()).device
        print(f"Model parameters are on device: {model_device}")

        input_ids = inputs["input_ids"].to(model_device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model_device)

        print(f"Input shape: {input_ids.shape}")

        # Hook to capture KV caches from the specified layer
        kv_caches = {}

        def capture_kv_hook(module, input, output):
            """Hook to capture attention KV caches"""
            try:
                # Handle new transformers cache format
                if isinstance(output, tuple) and len(output) >= 3:
                    # Item 2 should be the DynamicCache
                    cache = output[2]
                    if hasattr(cache, 'key_cache') and hasattr(cache, 'value_cache'):
                        # Get the current layer's KV from cache
                        if len(cache.key_cache) > 0 and len(cache.value_cache) > 0:
                            key = cache.key_cache[0]  # First layer
                            value = cache.value_cache[0]  # First layer
                            kv_caches['key'] = key.detach().cpu()
                            kv_caches['value'] = value.detach().cpu()
                            print(f"Captured KV from DynamicCache: key shape {key.shape}, value shape {value.shape}")
                            return

                # Fallback: try old format
                if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                    present_kv = output[1]
                    if isinstance(present_kv, tuple) and len(present_kv) >= 2:
                        key, value = present_kv[0], present_kv[1]
                        kv_caches['key'] = key.detach().cpu()
                        kv_caches['value'] = value.detach().cpu()
                        print(f"Captured KV in old format: key shape {key.shape}, value shape {value.shape}")
                        return

                # Debug output
                print(f"Hook received output type: {type(output)}")
                if isinstance(output, tuple):
                    print(f"Output tuple length: {len(output)}")
                    for i, item in enumerate(output):
                        if hasattr(item, 'shape'):
                            print(f"  Item {i} shape: {item.shape}")
                        else:
                            print(f"  Item {i} type: {type(item)}")

            except Exception as e:
                print(f"Error in KV hook: {e}")

        # Register hook on the specified layer
        target_layer = self.model.model.layers[layer_idx].self_attn
        hook_handle = target_layer.register_forward_hook(capture_kv_hook)

        try:
            with torch.no_grad():
                # Forward pass to trigger hook
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    output_attentions=False,
                    use_cache=True
                )

            # Check if we captured KV caches
            if 'key' not in kv_caches or 'value' not in kv_caches:
                raise ValueError("Failed to capture KV caches from model")

            keys = kv_caches['key']
            values = kv_caches['value']

            print(f"Captured KV: keys shape {keys.shape}, values shape {values.shape}")

            # Convert to numpy and reshape
            # Original shape: [batch_size, num_heads, seq_len, head_dim]
            # We want: [seq_len, hidden_size]
            batch_size, num_heads, seq_len, head_dim = keys.shape

            # Concatenate heads along the last dimension
            keys_flat = keys.reshape(batch_size, seq_len, -1).squeeze(0)  # [seq_len, hidden_size]
            values_flat = values.reshape(batch_size, seq_len, -1).squeeze(0)  # [seq_len, hidden_size]

            # Convert to numpy
            keys_np = keys_flat.numpy().astype(np.float32)
            values_np = values_flat.numpy().astype(np.float32)

            print(f"Final shapes: keys {keys_np.shape}, values {values_np.shape}")
            return keys_np, values_np

        finally:
            # Always remove the hook
            hook_handle.remove()

    def extract_multi_layer_kv(self, text: str, max_length: int = 2048) -> dict:
        """
        Extract KV vectors from multiple layers.

        Returns:
            Dict with layer_idx -> (keys, values) mapping
        """
        results = {}
        for layer_idx in range(min(4, self.num_layers)):  # Extract from first 4 layers
            try:
                keys, values = self.extract_kv_from_text(text, max_length, layer_idx)
                results[layer_idx] = (keys, values)
                print(f"Layer {layer_idx}: extracted {keys.shape[0]} KV pairs")
            except Exception as e:
                print(f"Failed to extract from layer {layer_idx}: {e}")
                continue

        return results
    def cleanup(self):
        """Cleanup method for compatibility"""
        pass

def run_real_model_comparison():
    """
    Run a complete experiment comparing MP-KVM with real model data.
    """
    print("Running MP-KVM experiment with real Llama model data...")

    # Setup model
    model_path = "model/Llama-3.1-8B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        model, tokenizer = setup_model_and_tokenizer(model_path, device)
        extractor = RealModelKVExtractor(model, tokenizer, device)

        # Create long context
        context_text = create_long_context_text(4000)  # 4000 tokens worth of text
        print(f"Created context text with ~{len(context_text)//4} tokens")

        # Extract KV from first layer
        keys, values = extractor.extract_kv_from_text(context_text, max_length=1024, layer_idx=0)

        print(f"Extracted {keys.shape[0]} KV pairs from real model")
        print(f"Key dimension: {keys.shape[1]}")

        # Here you would run your MP-KVM compression experiments
        # For now, just demonstrate the extraction works

        return keys, values

    except Exception as e:
        print(f"Error in real model experiment: {e}")
        raise


if __name__ == "__main__":
    # Quick test
    keys, values = run_real_model_comparison()
    print(f"Success! Extracted {keys.shape[0]} real KV pairs")