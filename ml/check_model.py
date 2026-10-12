
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Name of the pretrained DistilBERT model
model_name = "distilbert/distilbert-base-uncased"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load DistilBERT with a classification head for 6 emotions
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=6
)

# Test sentence
text = "I am feeling happy today!"

# Convert text into model inputs
inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True
)

print("Model loaded successfully!")
print("Input IDs:", inputs["input_ids"])
print("Attention Mask:", inputs["attention_mask"])
print("Number of emotion classes:", model.config.num_labels)

