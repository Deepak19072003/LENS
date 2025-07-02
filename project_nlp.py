import streamlit as st
import joblib
import pandas as pd 

# --- Custom CSS with animation ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom, rgba(0,0,0,0.85), rgba(20,20,20,0.9));
        background-attachment: fixed;
        background-position: center;
        color: white;
        transition: background 0.5s ease-in-out;
    }

    .block-container {
        background-color: rgba(0, 0, 0, 0.65);
        padding: 2rem;
        border-radius: 12px;
        transition: background-color 0.5s ease-in-out;
    }

    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 8px 20px;
        font-size: 14px;
        border-radius: 6px;
        margin-top: 10px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        transition: background-color 0.3s ease, transform 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #45a049;
        cursor: pointer;
        transform: scale(1.05);
    }

    section[data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.85);
    }

    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        color: #ccc;
        padding: 10px;
        font-size: 13px;
    }

    .main-header {
        padding: 1rem;
        border-radius: 8px;
        background: linear-gradient(to right, #00c6ff, #0072ff);
        text-align: center;
        font-weight: bold;
        font-size: 26px;
        color: white;
        animation: slideIn 1s ease-out;
    }

    .subtitle {
        text-align: center;
        font-size: 16px;
        margin-bottom: 20px;
        color: #ddd;
        animation: fadeIn 2s ease-in;
    }

    @keyframes slideIn {
        from { transform: translateY(-50px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    .robot-animation {
        text-align: center;
        margin-bottom: 20px;
    }

    .robot-animation img {
        width: 140px;
        animation: float 3s ease-in-out infinite;
    }

    @keyframes float {
        0% { transform: translatey(0px); }
        50% { transform: translatey(-10px); }
        100% { transform: translatey(0px); }
    }
    </style>
""", unsafe_allow_html=True)

# --- Load models ---
spam_model = joblib.load("spam_classifier.pkl")
language_model = joblib.load("lang_det.pkl")
news_model = joblib.load("news_cat.pkl")
review_model = joblib.load("review.pkl")

# --- Page Setup ---
st.set_page_config(page_title="LENSE eXpert", layout="wide")

# --- Sidebar ---
st.sidebar.image("Title_image1.jpg", use_container_width=True)
st.sidebar.title("🔍LENSE eXpert")

# Tool selection dropdown
tool = st.sidebar.selectbox("🔍 Choose a Tool", (
    "📩 Spam Classifier",
    "🌐 Language Detection",
    "🍔 Review Sentiment",
    "📰 News Classification"
))

# Sidebar expandable sections
with st.sidebar.expander("ℹ️ About Us"):
    st.write("LENSE eXpert harnesses cutting-edge Machine Learning and NLP to deliver fast, smart, and user-friendly text analysis — your go-to AI partner for insightful classification and understanding.")

with st.sidebar.expander("🎯 Mission"):
    st.write("Deliver accessible and effective AI-powered NLP solutions for developers, businesses, and learners alike.")

with st.sidebar.expander("📞 Contact Us"):
    st.write("📧 deepakrastogi4546@gmail.com")
    st.write("📱 +91 6398892809")

# --- Header and Intro ---
st.markdown('<div class="main-header">🔍 LENSE eXpert - NLP Suite</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">✨ Use powerful AI tools to analyze and classify text in seconds ✨</div>', unsafe_allow_html=True)

# --- Tool Logic ---
if tool == "📩 Spam Classifier":
    st.subheader("📩 Spam Detection – Check if a message is spam or not")
    msg = st.text_input("✉️ Enter a message:", placeholder="e.g., Free prize claim now!")
    if st.button("🚀 Predict Spam"):
        if msg.strip() == "":
            st.warning("⚠️ Please enter a message before predicting.")
        else:
            prediction = spam_model.predict([msg])[0]
            result = "🛑 Spam" if prediction == 0 else "✅ Not Spam"
            st.success(f"Prediction: {result}")

    st.markdown("### 📁 Bulk Prediction")
    file = st.file_uploader("Upload .txt or .csv file", type=["txt", "csv"])
    if file:
        df = pd.read_csv(file, header=None, names=["Message"])
        df["Prediction"] = spam_model.predict(df.Message)
        df["Prediction"] = df["Prediction"].map({0: "Spam", 1: "Not Spam"})
        st.dataframe(df)

elif tool == "🌐 Language Detection":
    st.subheader("🌐 Language Detection – Guess the language of any sentence!")
    text = st.text_input("🗣️ Enter a sentence:", placeholder="e.g., Bonjour tout le monde")
    if st.button("🌍 Detect Language"):
        if text.strip() == "":
            st.warning("⚠️ Please enter a sentence before detecting language.")
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
    review = st.text_input("📝 Enter a review:", placeholder="e.g., The food was absolutely delicious!")
    if st.button("🔍 Analyze Review"):
        if review.strip() == "":
            st.warning("⚠️ Please enter a review before analyzing.")
        else:
            prediction = review_model.predict([review])[0]
            label = "😋 Delicious!" if prediction == 1 else "😐 Not impressive"
            image_path = "Liked.jpg" if prediction == 1 else "Not Liked.jpg"
            st.success(f"Review Sentiment: {label}")
            st.image(image_path, width=160)

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
        df["Prediction"] = df["Prediction"].map({0: "😐 Not impressive", 1: "😋 Delicious!"})
        st.dataframe(df)

elif tool == "📰 News Classification":
    st.subheader("📰 News Classification – Categorize your headlines smartly")
    headline = st.text_input("📰 Enter headline:", placeholder="e.g., Government announces new policy reforms")
    if st.button("🧠 Classify News"):
        if headline.strip() == "":
            st.warning("⚠️ Please enter a headline before classifying.")
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
        © 2025 LENSE eXpert. All rights reserved 
        👨 Made with ❤️ by <b>Deepak Rastogi
    </div>
''', unsafe_allow_html=True)

