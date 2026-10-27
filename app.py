import html
import textwrap
from pathlib import Path

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Mood.ly | Emotion Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# DESIGN SYSTEM
# =========================================================
st.markdown(
    textwrap.dedent("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@400;500;600;700&display=swap');

    :root {
        --cream: #F8F6EF;
        --paper: #FFFDF8;
        --forest: #28483B;
        --sage: #66836B;
        --sage-light: #EAF0E6;
        --text: #40574D;
        --muted: #7E8C82;
        --border: #E5E3D8;
    }

    .stApp {
        background: var(--cream);
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }
    header[data-testid="stHeader"] { background: transparent; }
    #MainMenu, footer { visibility: hidden; }
    .block-container { max-width: 1120px; padding: 1.5rem 2rem 2rem; }

    h1, h2, h3 {
        font-family: 'Playfair Display', Georgia, serif !important;
        color: var(--forest) !important;
    }
    p { color: var(--text); font-family: 'DM Sans', sans-serif; }

    .topbar {
        display:flex; align-items:center; justify-content:space-between;
        border-bottom:1px solid var(--border); padding-bottom:1.2rem;
        margin-bottom:3rem;
    }
    .brand {
        color:var(--forest); font-family:'Playfair Display',Georgia,serif;
        font-size:2.1rem; font-weight:600; letter-spacing:-1px;
    }
    .brand span { margin-right:.4rem; }
    .top-note {
        color:var(--muted); font-size:.75rem; letter-spacing:.13em;
        text-transform:uppercase;
    }

    .eyebrow {
        color:var(--sage); font-size:.75rem; font-weight:700;
        letter-spacing:.22em; text-transform:uppercase; margin-bottom:1rem;
    }
    .hero-title {
        color:var(--forest); font-family:'Playfair Display',Georgia,serif;
        font-size:clamp(2.7rem,5vw,4.5rem); font-weight:500;
        letter-spacing:-.045em; line-height:1.08; margin-bottom:1.2rem;
    }
    .hero-title em { color:var(--sage); font-weight:400; }
    .hero-description {
        color:#718176; font-size:1rem; line-height:1.9; max-width:520px;
    }
    .hero-art {
        display:flex; min-height:250px; justify-content:center; align-items:center;
        text-align:center; border-radius:48% 52% 45% 55%;
        background:radial-gradient(ellipse at center,#E9ECDD 0%,#F0F0E5 48%,rgba(248,246,239,0) 75%);
    }
    .art-leaf { font-size:4.8rem; }
    .art-quote {
        color:#567361; font-family:'Playfair Display',Georgia,serif;
        font-size:1.15rem; font-style:italic; line-height:1.9;
    }


    div[data-testid="stWidgetLabel"] p {
        color:var(--forest) !important; font-weight:600 !important; font-size:.9rem !important;
    }
    .stTextArea textarea {
        background:#FFFFFF !important; color:var(--text) !important;
        border:1px solid #D6DED2 !important; border-radius:12px !important;
        padding:1rem !important; font-family:'DM Sans',sans-serif !important;
        font-size:1rem !important; line-height:1.8 !important; box-shadow:none !important;
    }
    .stTextArea textarea::placeholder { color:#9AA69C !important; opacity:1 !important; }
    .stTextArea textarea:focus {
        border:1px solid var(--sage) !important;
        box-shadow:0 0 0 3px rgba(102,131,107,.12) !important; outline:none !important;
    }
    .stButton > button, .stFormSubmitButton > button {
        background:var(--forest) !important; color:#FFFFFF !important;
        border:1px solid var(--forest) !important; border-radius:11px !important;
        min-height:3.2rem; width:100%; font-family:'DM Sans',sans-serif !important;
        font-weight:600 !important; font-size:.98rem !important; transition:all .2s ease;
    }
    .stButton > button p, .stFormSubmitButton > button p { color:#FFFFFF !important; }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background:#466750 !important; border-color:#466750 !important; transform:translateY(-1px);
    }

    .result-panel {
        background:#EDF1E8; border:1px solid #DCE4D6; border-radius:20px;
        padding:2.2rem 1.5rem; text-align:center; margin-top:2rem;
    }
    .result-eyebrow {
        color:var(--sage); font-size:.73rem; font-weight:700;
        letter-spacing:.2em; margin-bottom:1.2rem; text-transform:uppercase;
    }
    .result-emoji { font-size:3.5rem; margin-bottom:.5rem; }
    .result-title {
        color:var(--forest); font-family:'Playfair Display',Georgia,serif;
        font-size:2.6rem; font-weight:600; margin-bottom:.6rem;
    }
    .result-description {
        color:#687C6D; font-size:.95rem; line-height:1.8; max-width:560px; margin:0 auto;
    }
    .confidence-number { color:var(--forest); font-size:1.5rem; font-weight:700; margin-top:1.3rem; }
    .confidence-label { color:var(--muted); font-size:.8rem; margin-top:.2rem; }

    .score-heading {
        color:var(--forest); font-family:'Playfair Display',Georgia,serif;
        font-size:1.8rem; margin:2rem 0 .4rem;
    }
    .score-subtitle { color:var(--muted); font-size:.9rem; margin-bottom:1.4rem; }
    .score-card {
        background:var(--paper); border:1px solid var(--border); border-radius:13px;
        padding:1rem; margin-bottom:.8rem;
    }
    .score-header {
        display:flex; align-items:center; justify-content:space-between;
        gap:.5rem; margin-bottom:.7rem;
    }
    .score-name { color:var(--forest); font-size:.92rem; font-weight:600; }
    .score-value { color:var(--muted); font-size:.85rem; font-weight:600; }
    .score-track { width:100%; height:7px; background:#E8EBE3; border-radius:20px; overflow:hidden; }
    .score-fill { height:100%; border-radius:20px; }

    .info-card {
        background:var(--paper); border:1px solid var(--border);
        border-radius:16px; padding:1.4rem; min-height:190px;
    }
    .info-icon { font-size:1.7rem; margin-bottom:.7rem; }
    .info-title { color:var(--forest); font-weight:700; font-size:.95rem; margin-bottom:.5rem; }
    .info-text { color:var(--muted); font-size:.85rem; line-height:1.8; }

    .footer {
        border-top:1px solid var(--border); color:var(--muted); font-size:.8rem;
        line-height:1.9; margin-top:3rem; padding-top:1.5rem; text-align:center;
    }
    .footer strong { color:var(--forest); }

    @media (max-width:700px) {
        .block-container { padding:1rem 1rem 2rem; }
        .topbar { margin-bottom:1.8rem; }
        .brand { font-size:1.7rem; }
        .top-note { font-size:.6rem; }
        .hero-title { font-size:2.7rem; }
        .hero-art { min-height:180px; }
        .writing-card { padding:1rem; }
        .result-title { font-size:2rem; }
    }
    </style>
    """),
    unsafe_allow_html=True,
)

# =========================================================
# EMOTION LABELS
# These IDs must match the label mapping used during training.
# =========================================================
EMOTIONS = {
    0: {
        "name": "Sadness", "emoji": "🌧️", "color": "#668BC4",
        "description": "Your words may reflect sadness, disappointment, or emotional heaviness.",
    },
    1: {
        "name": "Joy", "emoji": "☀️", "color": "#D8A83F",
        "description": "Your words express happiness, positivity, or a sense of accomplishment.",
    },
    2: {
        "name": "Love", "emoji": "💗", "color": "#D77F9B",
        "description": "Your words may express affection, warmth, or a sense of connection.",
    },
    3: {
        "name": "Anger", "emoji": "🔥", "color": "#CF6D61",
        "description": "Your words may express frustration, irritation, or displeasure.",
    },
    4: {
        "name": "Fear", "emoji": "🌫️", "color": "#8B79B9",
        "description": "Your words may reflect worry, uncertainty, or apprehension.",
    },
    5: {
        "name": "Surprise", "emoji": "✨", "color": "#4D9FA1",
        "description": "Your words may express astonishment or an unexpected reaction.",
    },
}


# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_emotion_model():
    model_path = "KeerthanaB2005/mood.ly-emotion-model"

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.eval()
    return tokenizer, model


# =========================================================
# PREDICTION
# =========================================================
def predict_emotion(text):
    tokenizer, model = load_emotion_model()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=96,
    )
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    predicted_id = int(torch.argmax(probabilities).item())
    return predicted_id, probabilities


# =========================================================
# HEADER
# =========================================================
st.markdown(
    textwrap.dedent("""
    <div class="topbar">
        <div class="brand"><span>🌿</span>Mood.ly</div>
        <div class="top-note">A little space for yourself ♡</div>
    </div>
    """),
    unsafe_allow_html=True,
)

# =========================================================
# HERO
# =========================================================
left, right = st.columns([1.2, 0.8], gap="large", vertical_alignment="center")

with left:
    st.markdown(
        textwrap.dedent("""
        <div class="eyebrow">Your AI emotion companion</div>
        <div class="hero-title">Your words<br>have <em>feelings</em> too.</div>
        <div class="hero-description">
            Every sentence carries a little piece of your world.
            Put your thoughts into words and let Mood.ly explore
            the emotion expressed in your writing.
        </div>
        """),
        unsafe_allow_html=True,
    )

with right:
    st.markdown(
        textwrap.dedent("""
        <div class="hero-art">
            <div>
                <div class="art-leaf">🌿</div>
                <div class="art-quote">
                    Same words.<br>
                    Different emotions.<br>
                    Let's find yours. ♡
                </div>
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )

# =========================================================
# INPUT — no "Your thoughts" or "Choose an example" sections
# =========================================================

with st.form("moodly_form", clear_on_submit=False):
    user_text = st.text_area(
        "Message",
        placeholder="I've been working hard, and today I'm proud of myself...",
        height=160,
        max_chars=2000,
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("✦  Discover my emotion", use_container_width=True)

# =========================================================
# RESULTS
# =========================================================
if submitted:
    cleaned_text = user_text.strip()

    if not cleaned_text:
        st.warning("Please enter a sentence before starting the analysis.")
    else:
        try:
            with st.spinner("Exploring the emotion in your words..."):
                predicted_id, probabilities = predict_emotion(cleaned_text)

            emotion = EMOTIONS[predicted_id]
            confidence = float(probabilities[predicted_id].item())

            # dedent is important: it prevents Markdown from treating the
            # indented HTML as a code block and showing the tags literally.
            result_html = textwrap.dedent(f"""
            <div class="result-panel">
                <div class="result-eyebrow">Your emotion insight</div>
                <div class="result-emoji">{emotion["emoji"]}</div>
                <div class="result-title">{html.escape(emotion["name"])}</div>
                <div class="result-description">{html.escape(emotion["description"])}</div>
                <div class="confidence-number">{confidence:.1%}</div>
                <div class="confidence-label">Model score for the predicted emotion</div>
            </div>
            """)
            st.markdown(result_html, unsafe_allow_html=True)

            st.markdown(
                textwrap.dedent("""
                <div class="score-heading">The emotional landscape</div>
                <div class="score-subtitle">
                    A breakdown of the model's scores across six emotions.
                </div>
                """),
                unsafe_allow_html=True,
            )

            left_scores, right_scores = st.columns(2, gap="medium")

            for emotion_id, details in EMOTIONS.items():
                score = float(probabilities[emotion_id].item())
                percentage = score * 100
                score_html = textwrap.dedent(f"""
                <div class="score-card">
                    <div class="score-header">
                        <span class="score-name">{details["emoji"]} &nbsp; {html.escape(details["name"])}</span>
                        <span class="score-value">{percentage:.1f}%</span>
                    </div>
                    <div class="score-track">
                        <div class="score-fill" style="width:{percentage:.4f}%; background:{details["color"]};"></div>
                    </div>
                </div>
                """)

                target_column = left_scores if emotion_id % 2 == 0 else right_scores
                with target_column:
                    st.markdown(score_html, unsafe_allow_html=True)

        except Exception as error:
            st.error("Mood.ly couldn't analyze your text. Please check that the trained model is set up correctly.")
            with st.expander("Technical details"):
                st.exception(error)

# =========================================================
# ABOUT MOOD.LY — no "A gentle reminder" section
# =========================================================
st.markdown("<br>", unsafe_allow_html=True)
info2, info3 = st.columns(2, gap="medium")


with info2:
    st.markdown(
        textwrap.dedent("""
        <div class="info-card">
            <div class="info-icon">🌱</div>
            <div class="info-title">Made for reflection</div>
            <div class="info-text">
                A calm, simple interface designed to explore the emotional tone expressed through words.
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )

with info3:
    st.markdown(
        textwrap.dedent("""
        <div class="info-card">
            <div class="info-icon">✧</div>
            <div class="info-title">Your words, your story</div>
            <div class="info-text">
                An AI prediction is only one interpretation and cannot capture the full complexity of a person's feelings.
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    textwrap.dedent("""
    <div class="footer">
        <strong>🌿 Mood.ly</strong> &nbsp; | &nbsp; Text Emotion Classification
        <br>
        A student machine learning project, built with care.
    </div>
    """),
    unsafe_allow_html=True,
)