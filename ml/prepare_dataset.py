from datasets import load_dataset
from transformers import AutoTokenizer

# 1. Load the emotion dataset
dataset = load_dataset("dair-ai/emotion")

# 2. Load the DistilBERT tokenizer
model_name = "distilbert/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 3. Function to tokenize the text
def tokenize_data(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# 4. Tokenize all dataset splits
tokenized_dataset = dataset.map(
    tokenize_data,
    batched=True
)

# 5. Display the dataset structure
print("\nTokenized dataset:")
print(tokenized_dataset)

# 6. Display one sample
print("\nFirst training sample:")
print(tokenized_dataset["train"][0])

# 7. Display the label
print("\nEmotion label:", tokenized_dataset["train"][0]["label"])

print("\nDataset preparation completed!")

