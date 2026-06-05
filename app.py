# app.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# Path to your fine-tuned model
MODEL_DIR = "./amharic_qa_model"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_DIR)

# Create QA pipeline (requires transformers >= 4.30)
qa_pipeline = pipeline("table-question-answering", model=model, tokenizer=tokenizer)

# Streamlit UI
st.title("Amharic Question Answering Demo")

context = st.text_area("Enter context (Amharic text):")
question = st.text_input("Enter your question (Amharic):")

top_k = st.slider("Number of candidate answers", 1, 5, 1)

if st.button("Get Answer"):
    if context and question:
        results = qa_pipeline({"context": context, "question": question}, top_k=top_k)
        if isinstance(results, list):
            for i, res in enumerate(results):
                st.write(f"Answer {i+1}: {res['answer']} (confidence: {res['score']:.4f})")
        else:
            st.write("Answer:", results["answer"])
            st.write("Confidence:", results["score"])
