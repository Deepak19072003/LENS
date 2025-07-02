import streamlit as st
import joblib
import pandas as pd

# --- Load models ---
spam_model = joblib.load("spam_classifier.pkl")
language_model = joblib.load("lang_det.pkl")
news_model = joblib.load("news_cat.pkl")
review_model = joblib.load("review.pkl")

# --- Page Setup ---
st.set_page_config(page_title="LENS eXpert", layout="wide")

# --- Custom CSS ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom, rgba(0,0,0,0.85), rgba(20,20,20,0.9));
        background-attachment: fixed;
        color: white;
    }
    .block-container {
        background-color: rgba(0, 0, 0, 0.65);
        padding: 2rem;
        border-radius: 12px;
    }
    .tool-button {
        background-color: #4CAF50;
        color: white;
        padding: 0.4rem 1rem;
        border: none;
        border-radius: 6px;
        font-size: 14px;
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .tool-button:hover {
        background-color: #45a049;
        cursor: pointer;
    }
    .main-header {
        text-align: center;
        font-size: 26px;
        padding: 1rem;
        background: linear-gradient(to right, #00c6ff, #0072ff);
        border-radius: 8px;
        color: white;
    }
    .subtitle {
        text-align: center;
        font-size: 16px;
        color: #ccc;
        margin-bottom: 20px;
    }
    .footer {
        text-align: center;
        font-size: 13px;
        color: #ccc;
        margin-top: 40px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.image("Title_image1.jpg", use_container_width=True)
st.sidebar.title("🔍 LENS eXpert")

# --- Session state for tool navigation ---
if 'selected_tool' not in st.session_state:
    st.session_state.selected_tool = "📩 Spam Classifier"

if st.sidebar.button("📩 Spam Classifier"):
    st.session_state.selected_tool = "📩 Spam Classifier"
if st.sidebar.button("🌐 Language Detection"):
    st.session_state.selected_tool = "🌐 Language Detection"
if st.sidebar.button("🍔 Review Sentiment"):
    st.session_state.selected_tool = "🍔 Review Sentiment"
if st.sidebar.button("📰 News Classification"):
    st.session_state.selected_tool = "📰 News Classification"

# --- Sidebar Info ---
with st.sidebar.expander("ℹ️ More Info"):
    st.markdown("### About Us")
    st.write("LENS eXpert uses powerful NLP and AI models to classify and analyze text with ease.")
    st.markdown("### Mission")
    st.write("Make AI-powered tools accessible and helpful for students, developers, and businesses.")
    st.markdown("### Contact Us")
    st.write("📧 deepakrastogi4546@gmail.com")
    st.write("📱 +91 6398892809")

# --- Header ---
st.markdown('<div class="main-header">🔍 LENS eXpert (NLP Suites)</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">✨ Analyze and classify text using AI ✨</div>', unsafe_allow_html=True)

# --- Tool Logic ---
tool = st.session_state.selected_tool

if tool == "📩 Spam Classifier":
    st.subheader("📩 Spam Detection – Check if a message is spam or not")
    msg = st.text_input("✉️ Enter a message:")
    if st.button("🚀 Predict Spam"):
        if msg.strip() == "":
            st.warning("⚠️ Please enter a message.")
        else:
            prediction = spam_model.predict([msg])[0]
            result = "🛑 Spam" if prediction == 0 else "✅ Not Spam"
            st.success(f"Prediction: {result}")

    st.markdown("### 📁 Bulk Prediction")
    file = st.file_uploader("Upload a .txt or .csv file", type=["txt", "csv"])
    if file:
        df = pd.read_csv(file, header=None, names=["Message"])
        df["Prediction"] = spam_model.predict(df.Message)
        df["Prediction"] = df["Prediction"].map({0: "Spam", 1: "Not Spam"})
        st.dataframe(df)

elif tool == "🌐 Language Detection":
    st.subheader("🌐 Language Detection – Guess the language of any sentence!")
    text = st.text_input("🗣️ Enter a sentence:")
    if st.button("🌍 Detect Language"):
        if text.strip() == "":
            st.warning("⚠️ Please enter a sentence.")
        else:
            prediction = language_model.predict([text])[0]
            st.success(f"Detected Language: {prediction}")

    st.markdown("### 📁 Bulk Prediction")
    file = st.file_uploader("Upload .txt or .csv file", type=["txt", "csv"])
    if file:
        df = pd.read_csv(file, header=None, names=["Text"])
        df["Prediction"] = language_model.predict(df.Text)
        st.dataframe(df)

elif tool == "🍔 Review Sentiment":
    st.subheader("🍔 Restaurant Reviews – Understand sentiment behind food reviews")
    review = st.text_input("📝 Enter a review:")
    if st.button("🔍 Analyze Review"):
        if review.strip() == "":
            st.warning("⚠️ Please enter a review.")
        else:
            prediction = review_model.predict([review])[0]
            label = "😋 Positive" if prediction == 1 else "😐 Negative"
            image_path = "Liked.jpg" if prediction == 1 else "Not Liked.jpg"
            st.success(f"Review Sentiment: {label}")
            st.image(image_path, width=150)

            if prediction == 1:
                st.markdown("""
                <style>
                .balloon {
                    font-size: 40px;
                    animation: float 3s ease-in-out infinite;
                    position: relative;
                }
                @keyframes float {
                    0% { top: 0px; opacity: 1; }
                    50% { top: -30px; opacity: 0.7; }
                    100% { top: 0px; opacity: 1; }
                }
                .center {
                    text-align: center;
                    margin-top: 10px;
                }
                </style>
                <div class="center">
                    <div class="balloon">🎈 🎉 🥳 🎈</div>
                    <h4 style="color:lightgreen;">Cheers to great food and good vibes! 🥂🎈</h4>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("### 📁 Bulk Prediction")
    file = st.file_uploader("Upload .txt or .csv file", type=["txt", "csv"])
    if file:
        df = pd.read_csv(file, header=None, names=["Review"])
        df["Prediction"] = review_model.predict(df.Review)
        df["Prediction"] = df["Prediction"].map({0: "😐 Negative", 1: "😋 Positive"})
        st.dataframe(df)

elif tool == "📰 News Classification":
    st.subheader("📰 News Classification – Categorize your headlines smartly")
    headline = st.text_input("📰 Enter headline:")
    if st.button("🧠 Classify News"):
        if headline.strip() == "":
            st.warning("⚠️ Please enter a headline.")
        else:
            prediction = news_model.predict([headline])[0]
            st.success(f"Predicted Category: {prediction}")

    st.markdown("### 📁 Bulk Prediction")
    file = st.file_uploader("Upload .txt or .csv file", type=["txt", "csv"])
    if file:
        df = pd.read_csv(file, header=None, names=["Headline"])
        df["Prediction"] = news_model.predict(df.Headline)
        st.dataframe(df)

# --- Footer ---
st.markdown('''
    <div class="footer">
        © 2025 LENS eXpert. All rights reserved.<br>
        👨‍💻 Made with ❤️ by <b>Deepak Rastogi</b>
    </div>
''', unsafe_allow_html=True)
