import streamlit as st
import pickle, re
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
os.system("python -m spacy download en_core_web_sm")

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Customer Feedback Intelligence",
    page_icon="💬",
    layout="wide"
)

# ================= GLOBAL STYLE =================
st.markdown("""
<style>
.main-title {
    font-size:42px;
    font-weight:700;
    text-align:center;
    margin-bottom:5px;
}
.sub-title {
    text-align:center;
    color:#6c757d;
    margin-bottom:25px;
}
.kpi {
    background:#f8f9fa;
    padding:18px;
    border-radius:14px;
    text-align:center;
    box-shadow:0 4px 10px rgba(0,0,0,0.05);
}
.kpi h2 { margin:0; }
.badge-pos {
    background:#2ecc71;
    color:white;
    padding:6px 14px;
    border-radius:20px;
    font-weight:600;
}
.badge-neg {
    background:#e74c3c;
    color:white;
    padding:6px 14px;
    border-radius:20px;
    font-weight:600;
}
.footer {
    text-align:center;
    font-size:12px;
    color:gray;
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD NLP =================
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))

# ================= LOAD MODEL =================
model = pickle.load(open("models/sentiment_model.pkl", "rb"))
tfidf = pickle.load(open("models/tfidf.pkl", "rb"))

# ================= TOPIC KEYWORDS =================
TOPICS = {
    "Delivery": ["delivery", "late", "delay", "shipping"],
    "Price": ["price", "cost", "expensive", "cheap"],
    "Quality": ["quality", "defective", "broken", "damage"],
    "Support": ["support", "service", "help", "customer"]
}

# ================= FUNCTIONS =================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    doc = nlp(" ".join(tokens), disable=["parser", "ner"])
    return " ".join([t.lemma_ for t in doc])

def detect_topic(text):
    for topic, words in TOPICS.items():
        for w in words:
            if w in text:
                return topic
    return "Other"

# ================= SIDEBAR =================
st.sidebar.title("📌 About Project")
st.sidebar.write("""
**AI-Based Customer Feedback Intelligence System**

✔ Real-time sentiment analysis  
✔ Batch CSV processing  
✔ Topic detection  
✔ Smart alerting  
✔ Analytics dashboard  

**Tech Stack**
- NLP (NLTK + spaCy)
- TF-IDF
- Logistic Regression
- Streamlit
""")
st.sidebar.divider()
st.sidebar.write("👨‍💻 Built by **Ronak Gupta**")

# ================= HEADER =================
st.markdown("<div class='main-title'>💬 Customer Feedback Intelligence</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Turn customer feedback into actionable business insights</div>", unsafe_allow_html=True)

# ================= TABS =================
tab1, tab2, tab3 = st.tabs(
    ["🧠 Single Feedback", "📂 Batch Upload", "📊 Analytics Dashboard"]
)

# ================= TAB 1 =================
with tab1:
    feedback = st.text_area("Enter customer feedback", height=140)

    if st.button("Analyze Feedback"):
        cleaned = clean_text(feedback)
        vector = tfidf.transform([cleaned])
        pred = model.predict(vector)[0]
        prob = model.predict_proba(vector)[0]
        confidence = max(prob) * 100
        topic = detect_topic(cleaned)

        st.markdown("### Result")
        if pred == 1:
            st.markdown(f"<span class='badge-pos'>Positive</span> &nbsp; Topic: <b>{topic}</b>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span class='badge-neg'>Negative</span> &nbsp; Topic: <b>{topic}</b>", unsafe_allow_html=True)
            if confidence > 80:
                st.warning("🚨 High-confidence negative feedback")

        st.metric("Prediction Confidence", f"{confidence:.2f}%")

# ================= TAB 2 =================
with tab2:
    uploaded = st.file_uploader("Upload CSV (column name must be: review)", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded, engine="python")

        if "review" not in df.columns:
            st.error("CSV must contain column named 'review'")
        else:
            results = []
            for r in df["review"]:
                c = clean_text(r)
                v = tfidf.transform([c])
                p = model.predict(v)[0]
                prob = model.predict_proba(v)[0]
                conf = max(prob) * 100
                results.append({
                    "review": r,
                    "sentiment": "Positive" if p == 1 else "Negative",
                    "confidence": round(conf, 2),
                    "topic": detect_topic(c)
                })

            res = pd.DataFrame(results)
            st.session_state["data"] = res

            st.dataframe(res, use_container_width=True)
            st.download_button(
                "⬇️ Download Predictions",
                res.to_csv(index=False),
                "feedback_predictions.csv",
                "text/csv"
            )

# ================= TAB 3 =================
with tab3:
    if "data" not in st.session_state:
        st.info("Upload a CSV to view analytics")
    else:
        d = st.session_state["data"]
        total = len(d)
        neg = d[d["sentiment"] == "Negative"]
        neg_pct = (len(neg) / total) * 100

        # KPIs
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='kpi'><h2>{total}</h2><p>Total Reviews</p></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='kpi'><h2>{len(neg)}</h2><p>Negative Reviews</p></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='kpi'><h2>{neg_pct:.1f}%</h2><p>Negative %</p></div>", unsafe_allow_html=True)

        st.subheader("📊 Sentiment Distribution")
        fig1, ax1 = plt.subplots()
        d["sentiment"].value_counts().plot(kind="bar", ax=ax1)
        st.pyplot(fig1)

        st.subheader("📊 Top Negative Topics")
        if len(neg) > 0:
            fig2, ax2 = plt.subplots()
            neg["topic"].value_counts().plot(kind="bar", ax=ax2)
            st.pyplot(fig2)

            st.subheader("❌ Negative Reviews (Action Focus)")
            st.dataframe(neg, use_container_width=True)

# ================= FOOTER =================
st.divider()
st.markdown("<div class='footer'>Capstone Project • Built with NLP, ML & Streamlit</div>", unsafe_allow_html=True)
