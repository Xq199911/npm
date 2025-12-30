# Minimal synthetic dataset loaders for PoC experiments.
def load_longbench_subset(max_docs: int = 10, doc_len: int = 4096):
    """
    Return a small synthetic subset that mimics long documents.
    Each document is a repeated sentence to reach `doc_len` tokens (approx).
    """
    docs = []
    base = "In the beginning, the experiment sets a long context and observes model behavior. "
    for i in range(max_docs):
        # naive repetition to reach approximate length
        repeat = max(1, doc_len // len(base))
        docs.append((f"doc_{i}", base * repeat))
    return docs



