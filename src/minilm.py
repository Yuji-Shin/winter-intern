import torch
from sentence_transformers import SentenceTransformer, util

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

class MiniLMRetriever:
    def __init__(self, intent_bank):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(MODEL_NAME, device=self.device)
        self.intent_bank = intent_bank
        self.texts = [x["text"] for x in intent_bank]
        self.bank_emb = self.model.encode(self.texts, normalize_embeddings=True)

    def retrieve_topk(self, query: str, k: int = 3):
        q_emb = self.model.encode(query, normalize_embeddings=True)
        scores = util.cos_sim(q_emb, self.bank_emb)[0]
        topk = torch.topk(scores, k=min(k, len(self.intent_bank)))
        results = []
        for idx in topk.indices.tolist():
            results.append({
                "intent": self.intent_bank[idx]["intent"],
                "text": self.intent_bank[idx]["text"],
                "score": float(scores[idx]),
            })
        return results
