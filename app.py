import streamlit as st
import requests

st.title("✈ British Airways AI Customer Experience Dashboard")

review = st.text_area("Enter Customer Review:")

if st.button("Analyze Review"):
    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json={"review": review}
        )
        result = response.json()
        if response.status_code == 200 and "sentiment_score" in result:
            st.subheader("Analysis Results")
            st.write("Sentiment Score:", result["sentiment_score"])
            st.write("Predicted CSAT Score:", result["predicted_csat"])
            st.success("Recommendation: " + result["recommendation"])
        else:
            st.error(f"API Error: {result}")
    except Exception as e:
        st.error(f"Request failed: {e}")
