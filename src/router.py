"""
MiniLM acts as a candidate generator and returns the top intent candidates 
based on semantic similarity.

The router compares the candidate intent scores with a predefined threshold 
and determines the final intent.
"""
class IntentRouter:
    def __init__(self, retriever, threshold: float = 0.55): # you can change threshold here.
        self.retriever = retriever
        self.threshold = threshold

    def route(self, utterance: str): 
        candidates = self.retriever.retrieve_topk(utterance, k=3)
        best = candidates[0]
        if best["score"] >= self.threshold: # if score>threshold, choose intent.
            chosen = best
        else:
            chosen = {"intent": "UNKNOWN", "score": best["score"], "text": ""}
        return chosen, candidates
        # Candidates are returned to show which intents were considered, supporting feedback and analysis.
