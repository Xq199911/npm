"""
Plot generation metrics and Needles recall vs centroids for quick figures.
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt


def plot_generation_metrics(path_json="experiments/realmodel_out/generation_metrics.json", out_dir="experiments/realmodel_out"):
    with open(path_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = [d["prompt"] for d in data]
    bleu = [d["bleu4"] for d in data]
    rouge = [d["rougeL"] for d in data]
    x = range(len(prompts))
    plt.figure(figsize=(6,4))
    plt.bar(x, bleu, width=0.4, label="BLEU-4")
    plt.bar([i+0.4 for i in x], rouge, width=0.4, label="ROUGE-L")
    plt.xticks([i+0.2 for i in x], [p[:30]+"..." for p in prompts], rotation=20)
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(out_dir, "generation_metrics.png")
    plt.savefig(out, dpi=150)
    print("wrote", out)

def plot_needles_summary(path="experiments/needles_large_out/results.json", out_dir="experiments/needles_large_out"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    configs = [d["cfg"]["cluster_kwargs"] for d in data]
    cent = [d["num_centroids"] for d in data]
    recall = [d["recall"] for d in data]
    times = [d["total_time_s"] for d in data]
    # plot recall vs centroids
    plt.figure(figsize=(6,4))
    plt.scatter(cent, recall)
    for i,c in enumerate(configs):
        plt.text(cent[i], recall[i], f"init={c.get('init_preserve_first_n')}")
    plt.xlabel("num_centroids")
    plt.ylabel("recall")
    plt.tight_layout()
    out1 = os.path.join(out_dir, "recall_vs_centroids.png")
    plt.savefig(out1, dpi=150)
    print("wrote", out1)
    # plot time vs centroids
    plt.figure(figsize=(6,4))
    plt.scatter(cent, times)
    plt.xlabel("num_centroids")
    plt.ylabel("total_time_s")
    out2 = os.path.join(out_dir, "time_vs_centroids.png")
    plt.tight_layout()
    plt.savefig(out2, dpi=150)
    print("wrote", out2)

if __name__ == "__main__":
    plot_generation_metrics()
    plot_needles_summary()


