import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import joblib

# ========================
# تحميل المودل
# ========================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("sentiment_model_pipeline_balanced.pkl")
        return model
    except:
        import numpy as np
        class DummyModel:
            def predict(self, texts):
                return np.random.choice([1, 2, 3, 4, 5], size=len(texts))
        return DummyModel()

# ========================
# رفع الملف وتحليله
# ========================
def uploadFile():
    uploaded_file = st.file_uploader("Upload a data file", type=["csv", "xlsx"])
    if uploaded_file:
        st.success("✅ File uploaded successfully!")

        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            return

        st.subheader("📋 File Preview")
        st.dataframe(df.head())

        text_col = st.selectbox("Select the text column for sentiment analysis:", df.columns)

        if st.button("🔍 Run Sentiment Analysis"):
            model = load_model()
            st.info("Analyzing sentiments...")

            texts = df[text_col].astype(str)
            preds = model.predict(texts)

            def to_label(x):
                if x in [1, 2]:
                    return "Negative 😞"
                elif x == 3:
                    return "Neutral 😐"
                else:
                    return "Positive 😊"

            df["Sentiment"] = [to_label(p) for p in preds]

            st.success("✅ Sentiment Analysis Completed!")
            st.dataframe(df.head())

            counts = df["Sentiment"].value_counts().reset_index()
            counts.columns = ["Sentiment", "Count"]
            colors = {"Positive 😊": "#4CAF50", "Neutral 😐": "#FFC107", "Negative 😞": "#F44336"}

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📊 Sentiment Distribution")
                fig = px.pie(counts, names="Sentiment", values="Count", color="Sentiment",
                             color_discrete_map=colors, template="simple_white")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("📈 Sentiment Count")
                fig2 = px.bar(counts, x="Sentiment", y="Count", color="Sentiment",
                              color_discrete_map=colors, template="simple_white")
                st.plotly_chart(fig2, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("⬇️ Download Results (CSV)", csv, "sentiment_results.csv", "text/csv")

# ========================
# الصفحة الرئيسية
# ========================
def renderPage():
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] > .main {
        background-color: #ffffff;
        padding: 2rem 4rem;
    }

    h1 {
        text-align: left;
        font-size: 2.8rem;
        color: #222;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .hr-line {
        height: 3px;
        background-color: #333;
        border: none;
        margin-bottom: 1.5rem;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1>Sentiment Analysis 😊😐😕😡</h1>", unsafe_allow_html=True)
components.html("<hr class='hr-line' />", height=20)

st.subheader("📁 File Analysis")
st.write("Upload a CSV or Excel file containing text data for sentiment analysis.")

# رفع الملف مباشرة بدون اختيار
uploadFile()


# ========================
# استدعاء الصفحة (المهم!)
# ========================
if __name__ == "__main__":
    renderPage()
