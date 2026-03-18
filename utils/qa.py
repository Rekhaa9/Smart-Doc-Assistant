import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Global instance for efficiency
_tokenizer = None
_model = None

def _load_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        model_id = "deepset/roberta-base-squad2"
        _tokenizer = AutoTokenizer.from_pretrained(model_id)
        _model = AutoModelForQuestionAnswering.from_pretrained(model_id)
    return _tokenizer, _model

def answer_question(question, context, max_length=384, stride=128):
    """
    Answer a question from context using RoBERTa-base-SQuAD2.
    
    Args:
    question (str): The question.
    context (str): The context passage.
    max_length (int): Max input length.
    stride (int): Stride for long contexts.
    
    Returns:
    dict: {'answer': str, 'score': float, 'start': int, 'end': int}
    """
    tokenizer, model = _load_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    
    # Tokenize with truncation and stride for long contexts
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation="only_second",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=True
    ).to(device)
    
    offset_mapping = inputs.pop("offset_mapping")
    sequence_ids = inputs.sequence_ids()
    
    start_logits = []
    end_logits = []
    
    with torch.no_grad():
        for i in range(inputs.input_ids.shape[0]):
            batch = {k: v[i:i+1].clone() for k, v in inputs.items()}
            outputs = model(**batch)
            start_logits.append(outputs.start_logits)
            end_logits.append(outputs.end_logits)
    
    start_logits = torch.cat(start_logits)
    end_logits = torch.cat(end_logits)
    
    # Get best start/end indices
    start_idx = torch.argmax(start_logits)
    end_idx = torch.argmax(end_logits) + 1
    
    score = torch.softmax(start_logits[start_idx] + end_logits[end_idx - 1], dim=0).max().item()
    
    # Decode answer
    input_ids = inputs["input_ids"][0:start_idx+end_idx]
    answer = tokenizer.decode(input_ids, skip_special_tokens=True)
    
    return {
        "answer": answer,
        "score": score,
        "start": int(start_idx),
        "end": int(end_idx)
    }
