import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = os.getenv("ROOT_CLS_OUT_DIR", "models/root_classifier")

_tok = None
_model = None

def _load():
    global _tok, _model
    if _model is None:
        _tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        _model.eval()
        if torch.cuda.is_available():
            _model.cuda()

def predict_root(text: str) -> str:
    _load()
    if not text:
        return "computer science"

    inputs = _tok(text[:2000], truncation=True, max_length=256, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        logits = _model(**inputs).logits
    pred_id = int(torch.argmax(logits, dim=-1).item())

    # transformers 保存时会带 id2label
    id2label = _model.config.id2label
    if isinstance(id2label, dict):
        return id2label[str(pred_id)] if str(pred_id) in id2label else id2label[pred_id]
    return id2label[pred_id]