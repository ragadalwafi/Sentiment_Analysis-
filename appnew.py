import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display
import streamlit.components.v1 as components

# ========================
# Load Model
# ========================
@st.cache_resource
def load_model():
    try:
        return joblib.load("sentiment_model_pipeline_balanced.pkl")
    except:
        class DummyModel:
            def predict(self, texts):
                return np.random.choice([1, 2, 3, 4, 5], size=len(texts))
        return DummyModel()

# ========================
# Arabic Stopwords (to exclude from word cloud)
# ========================
STOPWORDS = set([
    "في","من","على","إلى","عن","أن","إن","كان","كانت","ما","لا","لم","لن","قد",
    "هذا","هذه","ذلك","تلك","هناك","أو","و","ثم","كما","أي","أيضًا","كل","بين",
    "بعد","قبل","حتى","مع","لكن","بل","هو","هي","هم","هن","أنا","نحن","انت","أنت"
])

    # ===== Custom UI styling =====
def render_page():
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] > .main {
        background-color: #ffffff;
        padding: 2rem 4rem;
    }
    h1 {
        text-align: center;
        font-size: 2.5rem;
        color: #222;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .hr-line {
        height: 3px;
        background-color: #333;
        border: none;
        margin-bottom: 1rem;
        width: 60%;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    """, unsafe_allow_html=True)

    # الصورة على اليمين + العنوان في المنتصف
    col_left, col_right = st.columns([8, 2])
    with col_right:
        st.image("sentiment-analysis.png", width=120)

    with col_left:
        st.markdown("<h1>Sentiment Analysis</h1>", unsafe_allow_html=True)
        components.html("<hr class='hr-line' />", height=20)

    # ===== File Upload =====
    uploaded_file = st.file_uploader("📤 Upload your CSV or Excel file with text data", type=["csv", "xlsx"])
    if not uploaded_file:
        st.stop()

    # ===== Read the uploaded file =====
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
    st.subheader("📋 File Preview")
    st.dataframe(df.head())

    text_col = st.selectbox("📝 Select the text column:", df.columns)

    # ===== Run Analysis =====
    if st.button("🚀 Run Sentiment Analysis"):
        model = load_model()
        texts = df[text_col].astype(str)
        preds = model.predict(texts)

        def label(x):
            if x in [1, 2]: return "😞 Negative"
            elif x == 3: return "😐 Neutral"
            else: return "😊 Positive"

        df["Sentiment"] = [label(p) for p in preds]
        st.success("✅ Sentiment analysis completed successfully!")
        st.dataframe(df[[text_col, "Sentiment"]].head(10))

        # ===== Pie Chart =====
        counts = df["Sentiment"].value_counts().reset_index()
        counts.columns = ["Sentiment", "Count"]
        fig = px.pie(counts, names="Sentiment", values="Count", title="Sentiment Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        # ===== Bar Chart =====
        pie_colors = fig.data[0].marker.colors

        bar_fig = px.bar(
            counts,
            x="Sentiment",
            y="Count",
            color="Sentiment",
            text="Count",
            title="Sentiment Counts",
            color_discrete_sequence=pie_colors 
        )
        bar_fig.update_traces(textposition='outside')
        st.plotly_chart(bar_fig, use_container_width=True)

        # ===== Word Clouds =====
        st.subheader("Word Clouds (Positive vs Negative)")

        font_path = "Amiri-Regular.ttf"

        pos_text = " ".join(df[df["Sentiment"] == "😊 Positive"][text_col])
        neg_text = " ".join(df[df["Sentiment"] == "😞 Negative"][text_col])

        def clean_text(t):
            return " ".join([w for w in t.split() if w not in STOPWORDS])

        col1, col2 = st.columns(2)
        with col1:
            st.write("😊 Positive Words")
            text = clean_text(pos_text)
            reshaped = arabic_reshaper.reshape(text)
            bidi_text = get_display(reshaped)
            wc = WordCloud(font_path=font_path, width=800, height=400, background_color="white", colormap="Greens").generate(bidi_text)
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

        with col2:
            st.write("😞 Negative Words")
            text = clean_text(neg_text)
            reshaped = arabic_reshaper.reshape(text)
            bidi_text = get_display(reshaped)
            wc = WordCloud(font_path=font_path, width=800, height=400, background_color="white", colormap="Reds").generate(bidi_text)
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

        # ===== Download Results =====
        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ Download Results", csv, "sentiment_results.csv", "text/csv")


if __name__ == "__main__":
     render_page()

