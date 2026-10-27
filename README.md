# Mood.ly
### Text Emotion Detection Using Deep Learning

Mood.ly is a machine learning-powered web application that identifies
the emotion expressed in a piece of text using a fine-tuned DistilBERT
model. It provides a predicted emotion along with confidence scores
to help users explore the emotional tone of their words.

---

##  Features

-  Text emotion classification using DistilBERT
-  Supports six emotion categories
-  Displays prediction confidence scores
-  Clean and user-friendly Streamlit interface
-  Real-time predictions for user-entered text
-  Input validation for empty submissions

##  Emotion Categories

Mood.ly classifies text into the following categories:

- Joy
- Sadness
- Love
- Anger
- Fear
- Surprise

## Tech Stack

- **Language:** Python
- **Machine Learning:** PyTorch
- **NLP Model:** DistilBERT
- **Model Framework:** Hugging Face Transformers
- **Dataset:** dair-ai/emotion
- **Web Application:** Streamlit
- **Version Control:** Git and GitHub

##  Model Performance

The fine-tuned model achieved approximately **91.3% accuracy**
on the 2,000-example test set used during evaluation.

Performance varies across emotion categories. Surprise was more
challenging for the model than several other categories.

##  Installation and Setup

### 1. Clone the repository

