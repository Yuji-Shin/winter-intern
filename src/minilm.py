import torch
from sentence_transformers import SentenceTransformer, util

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

class MiniLMRetriever:
    def __init__(self, intent_bank):
        self.device = "cuda" if torch.cuda.is_available() else "cpu" 
        self.model = SentenceTransformer(MODEL_NAME, device=self.device) # load model
        # Since MiniLM serves as the brain of the system, it is loaded into the SentenceTransformer via the MODEL_NAME parameter.
        # Internet access is required for the initial model download. Afterward, ensure that the MiniLM files are present in the local directory.
        self.intent_bank = intent_bank 
        self.texts = [x["text"] for x in intent_bank] # extract text from 
        self.bank_emb = self.model.encode(self.texts, normalize_embeddings=True) #data embbeding, normalization, using MiniLM via the encode method.

    def retrieve_topk(self, query: str, k: int = 3):
        q_emb = self.model.encode(query, normalize_embeddings=True) # User Query Embedding
        scores = util.cos_sim(q_emb, self.bank_emb)[0] # Cosine similarity comparison between q_emb and bank_emb
        topk = torch.topk(scores, k=min(k, len(self.intent_bank))) # pick top K
        results = []
        for idx in topk.indices.tolist():
            results.append({
                "intent": self.intent_bank[idx]["intent"],
                "text": self.intent_bank[idx]["text"],
                "score": float(scores[idx]),
            }) 
        return results # The top-k intents along with similar texts from the intent bank and their corresponding scores are shown.

