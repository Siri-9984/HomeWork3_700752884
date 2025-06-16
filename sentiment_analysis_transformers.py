from transformers import pipeline

# Load sentiment-analysis pipeline (uses PyTorch by default)
classifier = pipeline("sentiment-analysis")

# Input text
text = "Despite the high price, the performance of the new MacBook is outstanding."

# Run and print
result = classifier(text)[0]
print(f"Sentiment: {result['label']}")
print(f"Confidence Score: {result['score']:.4f}")
