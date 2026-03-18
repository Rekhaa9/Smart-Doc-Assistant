import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Global instance for efficiency (loads once)
_tokenizer = None
_model = None

def _load_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        model_id = "facebook/bart-large-cnn"
        _tokenizer = AutoTokenizer.from_pretrained(model_id)
        _model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return _tokenizer, _model

def get_summary(text, max_new_tokens=60, min_length=30):
    """
    Summarize text using BART-large-CNN.
    
    Args:
    text (str): Input text to summarize.
    max_new_tokens (int): Max tokens in summary.
    min_length (int): Min length of summary.
    
    Returns:
    str: Summarized text.
    """
    tokenizer, model = _load_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=1024
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_length=min_length,
            num_beams=4,
            early_stopping=True,
            length_penalty=2.0,
            no_repeat_ngram_size=3
        )
    
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary
