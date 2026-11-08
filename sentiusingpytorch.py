"""Lazy FinBERT inference; this project uses a pretrained model, not fine-tuning."""
from functools import lru_cache


@lru_cache(maxsize=1)
def load_model():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
    model.eval()
    return tokenizer, model, device


def get_sentiment_batch(headlines):
    import numpy as np
    import torch
    if not headlines:
        return np.empty(0)
    if not all(isinstance(h, str) and h.strip() for h in headlines):
        raise ValueError("Headlines must be nonempty strings")
    tokenizer, model, device = load_model()
    inputs = tokenizer(headlines, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k:v.to(device) for k,v in inputs.items()}
    with torch.inference_mode():
        probabilities = model(**inputs).logits.softmax(dim=-1)
    labels = {str(v).lower():int(k) for k,v in model.config.id2label.items()}
    return (probabilities[:,labels["positive"]] - probabilities[:,labels["negative"]]).cpu().numpy()


if __name__ == "__main__":
    print(get_sentiment_batch(["Company reports strong earnings", "Company warns of losses"]))
