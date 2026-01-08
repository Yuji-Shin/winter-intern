class IntentRouter:
    def __init__(self, retriever, threshold: float = 0.55):
        self.retriever = retriever
        self.threshold = threshold

    def route(self, utterance: str):
        candidates = self.retriever.retrieve_topk(utterance, k=3)
        best = candidates[0]
        if best["score"] >= self.threshold:
            chosen = best
        else:
            chosen = {"intent": "UNKNOWN", "score": best["score"], "text": ""}
        return chosen, candidates
