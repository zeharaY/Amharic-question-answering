# app.py
import streamlit as st
from huggingface_hub import InferenceClient

# ============================================
# 1. Configuration from Secrets
# ============================================
# Your Hugging Face model ID (set in secrets or hardcoded)
MODEL_ID = st.secrets.get("HF_MODEL_ID", "your-username/amharic-qa-model")
HF_TOKEN = st.secrets.get("HF_TOKEN", None)

if HF_TOKEN is None:
    st.error("❌ Missing Hugging Face API token. Please set `HF_TOKEN` in your Streamlit secrets.")
    st.stop()

client = InferenceClient(api_key=HF_TOKEN, provider="hf-inference")

# ============================================
# 2. Helper Function
# ============================================
def get_answer(question: str, context: str) -> str:
    try:
        response = client.question_answering(
            question=question,
            context=context,
            model=MODEL_ID,
        )
        return response.get("answer", "No answer found.")
    except Exception as e:
        st.error(f"Inference error: {e}")
        return ""

# ============================================
# 3. UI
# ============================================
st.set_page_config(page_title="Amharic QA", page_icon="📚")
st.title("📚 Amharic Question Answering")
st.markdown("**Ask questions about any Amharic text – the model extracts the answer.**")

col1, col2 = st.columns(2)

with col1:
    context = st.text_area(
        "📄 **Context** (document to search)",
        height=300,
        placeholder="Paste Amharic text here..."
    )

with col2:
    question = st.text_area(
        "❓ **Question** (in Amharic)",
        height=150,
        placeholder="e.g., የአማርኛ ቋንቋ የትኛው ቤተሰብ ነው?"
    )
    ask_button = st.button("🔍 Get Answer", type="primary", use_container_width=True)

if ask_button:
    if not context.strip():
        st.warning("Please provide a context.")
    elif not question.strip():
        st.warning("Please provide a question.")
    else:
        with st.spinner("Searching for answer..."):
            answer = get_answer(question, context)
        st.success("✅ **Answer:**")
        st.markdown(f"> {answer}")

st.markdown("---")
st.caption("Powered by Hugging Face Inference API and your fine‑tuned Amharic QA model.")
