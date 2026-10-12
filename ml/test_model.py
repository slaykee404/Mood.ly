

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 1. Load our trained model
model_path = "./saved_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# 2. Define the emotion labels
emotions = [
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise"
]

# 3. Test sentences
sentences = [
    "I am feeling wonderful today!",
    "I am so upset and disappointed.",
    "I am scared about my future.",
    "I am extremely angry right now.",
    "I love spending time with my family.",
    "Wow, I never expected this surprise!"
]

# 4. Predict emotions
model.eval()

for sentence in sentences:

    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=96
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)
    predicted_id = torch.argmax(probabilities, dim=1).item()

    emotion = emotions[predicted_id]
    confidence = probabilities[0][predicted_id].item() * 100

    print("\nText:", sentence)
    print("Predicted emotion:", emotion)
    print(f"Confidence: {confidence:.2f}%")

