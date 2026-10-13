from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# 1. Load the dataset
print("Loading dataset...")

dataset = load_dataset("dair-ai/emotion")

# Use a small subset for our first training run
train_data = dataset["train"].shuffle(seed=42).select(range(4000))
val_data = dataset["validation"]

# 2. Load the tokenizer
model_name = "distilbert/distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# 3. Convert text into tokens
def tokenize_data(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=96
    )

print("Tokenizing dataset...")

train_data = train_data.map(tokenize_data, batched=True)
val_data = val_data.map(tokenize_data, batched=True)

# Keep only the columns needed by the model
train_data = train_data.remove_columns(["text"])
val_data = val_data.remove_columns(["text"])

# 4. Load DistilBERT with six emotion classes
print("Loading DistilBERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=6
)

# 5. Define evaluation metrics
def compute_metrics(pred):
    labels = pred.label_ids
    predictions = np.argmax(pred.predictions, axis=1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted")
    }

# 6. Configure training
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=50,
    report_to="none",
    use_cpu=True
)

# 7. Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    compute_metrics=compute_metrics
)

print("Starting training...")

trainer.train()

# 8. Evaluate the model
print("\nEvaluating model...")

results = trainer.evaluate()

print("Evaluation results:", results)

# 9. Save the trained model and tokenizer
save_path = "./saved_model"

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"\nModel saved successfully at {save_path}")