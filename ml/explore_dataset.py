from datasets import load_dataset
from collections import Counter

# Use a non-GUI backend for Matplotlib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# Load the Emotion dataset
dataset = load_dataset("dair-ai/emotion")

# Get emotion names
emotion_names = dataset["train"].features["label"].names

# Count each emotion
label_counts = Counter(dataset["train"]["label"])

print("Emotion Distribution:")

for label, count in label_counts.items():
    print(emotion_names[label], ":", count)


# Create bar graph
plt.bar(
    [emotion_names[label] for label in label_counts.keys()],
    label_counts.values()
)

plt.xlabel("Emotion")
plt.ylabel("Number of Samples")
plt.title("Emotion Distribution in Training Dataset")
plt.xticks(rotation=45)
plt.tight_layout()

# Save graph instead of opening a window
plt.savefig("ml/emotion_distribution.png")

print("\nGraph saved successfully!")




print("\nSample texts:")

for i in range(5):
    text = dataset["train"][i]["text"]
    label = dataset["train"][i]["label"]

    print("\nText:", text)
    print("Emotion:", emotion_names[label])