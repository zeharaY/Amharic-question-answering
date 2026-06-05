import streamlit as st
import requests

API_URL = "https://api-inference.huggingface.co/models/zeharay/amharic-qa-demo-model"
headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

st.title("Amharic QA Demo (HF API)")

context = st.text_area("Enter context:")
question = st.text_input("Enter question:")

if st.button("Get Answer"):
    if context and question:
        result = query({"inputs": {"question": question, "context": context}})
        st.write("Answer:", result.get("answer", "No answer"))
