from transformers import BertTokenizer, BertForSequenceClassification
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {device}")

tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
model = model.to(device)
print("model loaded!")
def get_sentiment_batch(headlines):
    # tokenize all headlines at once
    inputs = tokenizer(
        headlines,
        return_tensors="pt",
        padding=True,        # pad shorter headlines to same length
        truncation=True,     # cut headlines longer than 512 tokens
        max_length=512
    )
    # move to GPU
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # run model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # get probabilities
    scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # return positive - negative for each headline
    return (scores[:, 0] - scores[:, 1]).cpu().numpy()

if __name__ == "__main__":
    # move your test headlines code here
    test_headlines = [
        "Apple reports record breaking quarterly earnings beating all estimates",
        "Tesla recalls 50000 vehicles due to safety concerns",
        "Microsoft announces layoffs amid economic uncertainty"
    ]
    results = get_sentiment_batch(test_headlines)
    for headline, score in zip(test_headlines, results):
        print(f"{score:.4f} | {headline}")
    import sys
    sys.exit(0)
