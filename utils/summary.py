from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def get_summary(text):
    max_len = 1024
    text = text[:max_len]
    summary = summarizer(text, max_length=150, min_length=40, do_sample=False)
    return summary[0]['summary_text']