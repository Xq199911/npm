"""
Run generation with MPKVM injection ON and OFF and save outputs for comparison.
"""
from __future__ import annotations
import json
import os
import numpy as np


def run_one(model_dir: str, out_dir: str, enable_injection: bool, repeat: int = 5):
    os.makedirs(out_dir, exist_ok=True)
    from transformers import LlamaForCausalLM, AutoTokenizer

    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from core.integration_clean import MPKVMManager
    from adapters.llama_adapter import attach_mpkvm_to_hf_llama

    # load model
    model = LlamaForCausalLM.from_pretrained(model_dir, torch_dtype="auto", local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=True)

    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    mgr = MPKVMManager(dim=hidden_size, num_layers=num_layers, cluster_kwargs={"max_centroids": 1024, "sliding_window_size": 16384})

    attach_mpkvm_to_hf_llama(model, mgr, head_mean=True, sample_stride=1, enable_injection=enable_injection)

    prompts = [
        "In 100 words, explain the significance of manifold partitioned KV memories.",
        "Briefly summarize how centroid compression can help long-context transformers.",
        "Explain online clustering for KV caches and why it's useful.",
        "Discuss challenges of position encoding when merging KV tokens.",
    ]
    import torch
    gens = []
    for _ in range(max(1, int(repeat))):
        for p in prompts:
            inputs = tokenizer(p, return_tensors="pt") if tokenizer is not None else {"input_ids": torch.tensor([[1,2,3,4]])}
            out = model.generate(**inputs, max_new_tokens=64)
            if tokenizer is not None:
                text = tokenizer.decode(out[0], skip_special_tokens=True)
            else:
                text = str(out)
           
            gens.append({"prompt": p, "generation": text})

    # flush
    if hasattr(mgr, "flush_all_to_cpu"):
        mgr.flush_all_to_cpu()
        pass

    # force compress buffers
    for l, c in mgr.layers.items():
        if hasattr(c, "_compress_oldest_batch"):
            # compress any buffered items
            if hasattr(c, "keys_buffer"):
                n = len(c.keys_buffer)
                if n > 0:
                    c._compress_oldest_batch(n)

    # save gens
    gen_path = os.path.join(out_dir, "generations.json")
    with open(gen_path, "w", encoding="utf-8") as f:
        json.dump({"generations": gens}, f, indent=2)

    # save centroids per layer
    summary = {}
    for li in range(num_layers):
        centroids, counts, weights = mgr.get_layer_centroids(li)
        layer_dir = os.path.join(out_dir, f"layer_{li}")
        os.makedirs(layer_dir, exist_ok=True)
        np.save(os.path.join(layer_dir, "centroids.npy"), centroids)
        np.save(os.path.join(layer_dir, "counts.npy"), counts)
        np.save(os.path.join(layer_dir, "weights.npy"), weights)
        summary[li] = {"centroids": int(centroids.shape[0]) if centroids is not None else 0, "sum_counts": float(counts.sum()) if counts is not None and counts.size>0 else 0.0}

    with open(os.path.join(out_dir, "centroids_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"layers": summary}, f, indent=2)
    return gen_path, os.path.join(out_dir, "centroids_summary.json")


def compare_texts(path_on: str, path_off: str):
    jo = json.load(open(path_on, "r", encoding="utf-8"))
    jf = json.load(open(path_off, "r", encoding="utf-8"))
    gens_o = [g["generation"] for g in jo.get("generations", [])]
    gens_f = [g["generation"] for g in jf.get("generations", [])]
    diffs = []
    for i, (a, b) in enumerate(zip(gens_o, gens_f)):
        if a != b:
            diffs.append({"idx": i, "on": a, "off": b})
    return {"n_on": len(gens_o), "n_off": len(gens_f), "n_diff": len(diffs), "diffs": diffs}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="model/Llama-3.1-8B-Instruct")
    p.add_argument("--out", type=str, default="results/real_model")
    p.add_argument("--repeat", type=int, default=5)
    args = p.parse_args()
    base = args.out
    on_dir = os.path.join(base, "inject_on")
    off_dir = os.path.join(base, "inject_off")
    print('running injection ON...') 
    run_one(args.model, on_dir, enable_injection=True, repeat=args.repeat)
    print('running injection OFF...')
    run_one(args.model, off_dir, enable_injection=False, repeat=args.repeat)
    summary = compare_texts(os.path.join(on_dir, "generations.json"), os.path.join(off_dir, "generations.json"))
    print('comparison summary:', summary)

