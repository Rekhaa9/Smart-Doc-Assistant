from transformers import pipeline

qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

def answer_question(text, question):
    result = qa_pipeline(question=question, context=text[:3000])
    return result['answer']