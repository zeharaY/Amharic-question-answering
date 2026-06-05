import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

MODEL_REPO = "zeharay/amharic-qa-demo-model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_REPO)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

st.title("Amharic Question Answering Demo")

context = st.text_area("Enter context (Amharic text):")
question = st.text_input("Enter your question (Amharic):")

if st.button("Get Answer"):
    if context and question:
        result = qa_pipeline({"context": context, "question": question})
        st.write("Answer:", result["answer"])
        st.write("Confidence:", result["score"])
