# import streamlit as st
# import torch
# from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# MODEL_REPO = "zeharay/amharic-qa-demo-model"  # your public HF repo

# # Load tokenizer (disable fast tokenizer to avoid Rust build issues)
# # tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, use_fast=False)
# tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, use_fast=False)

# model = AutoModelForQuestionAnswering.from_pretrained(MODEL_REPO)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# st.title("Amharic Question Answering Demo")

# context = st.text_area("Enter context (Amharic text):")
# question = st.text_input("Enter your question (Amharic):")

# if st.button("Get Answer"):
#     if context and question:
#         inputs = tokenizer(question, context, return_tensors="pt").to(device)

#         with torch.no_grad():
#             outputs = model(**inputs)
#             start_idx = torch.argmax(outputs.start_logits)
#             end_idx = torch.argmax(outputs.end_logits) + 1

#         answer_tokens = inputs.input_ids[0][start_idx:end_idx]
#         answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

#         st.write("Answer:", answer)


import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

MODEL_REPO = "zeharay/amharic-qa-demo-model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, use_fast=False)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_REPO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

st.title("Amharic Question Answering Demo")

context = st.text_area("Enter context (Amharic text):")
question = st.text_input("Enter your question (Amharic):")

if st.button("Get Answer"):
    if context and question:
        inputs = tokenizer(question, context, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            start_idx = torch.argmax(outputs.start_logits)
            end_idx = torch.argmax(outputs.end_logits) + 1
        answer_tokens = inputs.input_ids[0][start_idx:end_idx]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
        st.write("Answer:", answer)

