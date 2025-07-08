import streamlit as st
from utils.document_loader import read_pdf, read_txt
from utils.summary import get_summary
from utils.qa import answer_question
from utils.challenge import generate_questions

# Page configuration
st.set_page_config(page_title="Free Smart Assistant", layout="centered")
st.title("🧠 Smart DOC Assistant")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file:
    if uploaded_file.name.endswith(".pdf"):
        text = read_pdf(uploaded_file)
    else:
        text = read_txt(uploaded_file)

    st.subheader("📌 Document Summary")
    st.info(get_summary(text))

    mode = st.radio("Choose a mode:", ["Ask Anything", "Challenge Me"])

    if mode == "Ask Anything":
        question = st.text_input("Ask a question based on the document:")
        if question:
            answer = answer_question(text, question)
            st.success(answer)

    elif mode == "Challenge Me":
        st.subheader("🧠 Try answering these:")
        questions = generate_questions(text)
        for i, q in enumerate(questions, 1):
            user_answer = st.text_input(f"{i}. {q}")
            if user_answer:
                st.write("✅ Answer received! (Feedback not available in free mode)")

# Footer credit
st.markdown(
    """
    <hr style="margin-top: 50px;">
    <div style="text-align: center; font-size: 14px; color: gray;">
         Built by <strong>Rekha Kumari</strong>
    </div>
    """,
    unsafe_allow_html=True
)