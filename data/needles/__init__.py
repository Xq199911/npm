# "Needle in a haystack" experiment utilities for PoC.
def run_needle_experiment(num_docs: int = 50, doc_len: int = 2048, insert_needle_at: int = 1234):
    """
    Produce synthetic documents where a single 'needle' token sequence is embedded
    in one of the documents. Returns (docs, needle_location).
    """
    docs = []
    needle = "NEEDLE_TOKEN_SEQUENCE"
    for i in range(num_docs):
        body = " ".join(["alpha"] * (doc_len // 5))
        docs.append((f"doc_{i}", body))
    # pick a random doc and position to insert needle (deterministic for PoC)
    target_doc = num_docs // 2
    docs[target_doc] = (docs[target_doc][0], docs[target_doc][1] + " " + needle + " ")
    needle_location = (target_doc, insert_needle_at)
    return docs, needle_location



