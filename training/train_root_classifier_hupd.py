# training/train_root_classifier_hupd.py
import os
import sys
import random
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

MODEL_NAME = os.getenv("ROOT_CLS_BASE_MODEL", "distilroberta-base")
OUT_DIR = os.getenv("ROOT_CLS_OUT_DIR", "models/cpc_classifier")
SEED = int(os.getenv("SEED", "42"))

LOCAL_JSONL = os.getenv("LOCAL_JSONL", "/root/autodl-tmp/data/patent.jsonl")

TRAIN_SAMPLES = int(os.getenv("TRAIN_SAMPLES", "28000"))
EVAL_SAMPLES = int(os.getenv("EVAL_SAMPLES", "3000"))
MAX_LEN = int(os.getenv("MAX_LEN", "256"))

random.seed(SEED)
torch.manual_seed(SEED)


def build_text(ex):
    title = (ex.get("title") or "").strip()
    abstract = (ex.get("abstract") or "").strip()
    summary = (ex.get("summary") or "").strip()
    background = (ex.get("background") or "").strip()
    full_description = (ex.get("full_description") or "").strip()
    claims = (ex.get("claims") or "").strip()

    main = abstract
    if not main:
        main = summary
    if not main:
        main = background[:2000]
    if not main:
        main = full_description[:3000]
    if not main:
        main = claims[:2000]

    text = f"{title} {main}".strip()
    return text if text else str(ex)[:4000]


def normalize_cpc(label: str) -> str:
    """
    可选：把更细粒度 CPC 截到前4位/前3位，降低类别数。
    例如:
    A61K31422 -> A61K
    G06F1730  -> G06F

    你现在先用前4位最稳。
    """
    if not label:
        return ""
    label = label.strip().upper()
    if len(label) >= 3:
        return label[:3]
    return label


def main():
    # =======================
    # 1) Load local dataset
    # =======================
    if not os.path.exists(LOCAL_JSONL):
        raise FileNotFoundError(f"Local jsonl not found: {LOCAL_JSONL}")

    print(f"Loading local patent dataset from: {LOCAL_JSONL}")
    ds = load_dataset("json", data_files=LOCAL_JSONL, split="train")
    ds = ds.shuffle(seed=SEED)

    print(f"Raw dataset size: {len(ds)}")

    # =======================
    # 2) Filter invalid labels
    # =======================
    def has_valid_label(ex):
        label = normalize_cpc(ex.get("main_cpc_label", ""))
        return label != ""

    ds = ds.filter(has_valid_label)
    print(f"After filtering empty CPC labels: {len(ds)}")

    if len(ds) < 50:
        raise ValueError(f"Dataset too small after filtering: {len(ds)}")

    # =======================
    # 3) Build label space
    # =======================
    all_labels = [normalize_cpc(x) for x in ds["main_cpc_label"]]
    label_counter = Counter(all_labels)

    print("Top CPC labels:", label_counter.most_common(20))

    unique_labels = sorted(label_counter.keys())
    label2id = {lab: i for i, lab in enumerate(unique_labels)}
    id2label = {i: lab for lab, i in label2id.items()}

    print(f"Num labels: {len(unique_labels)}")

    # =======================
    # 4) Add text + label id
    # =======================
    def add_fields(ex):
        cpc = normalize_cpc(ex.get("main_cpc_label", ""))
        ex["input_text"] = build_text(ex)
        ex["label"] = label2id[cpc]
        ex["cpc_norm"] = cpc
        return ex

    ds = ds.map(add_fields, num_proc=1)

    # =======================
    # 5) Train/eval split
    # =======================
    n = len(ds)

    req_train = TRAIN_SAMPLES
    req_eval = EVAL_SAMPLES

    max_train = int(n * 0.9)
    train_size = min(req_train, max_train)
    eval_size = min(req_eval, n - train_size)

    if eval_size < 20 and n >= 200:
        eval_size = max(20, int(n * 0.1))
        train_size = n - eval_size

    print(f"Using train_size={train_size}, eval_size={eval_size}")

    train_ds = ds.select(range(0, train_size))
    eval_ds = ds.select(range(train_size, train_size + eval_size))

    print("Train label dist:", Counter(train_ds["cpc_norm"]).most_common(20))
    print("Eval label dist :", Counter(eval_ds["cpc_norm"]).most_common(20))

    # =======================
    # 6) Tokenizer
    # =======================
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tok_fn(ex):
        return tok(ex["input_text"], truncation=True, max_length=MAX_LEN)

    train_tok = train_ds.map(tok_fn, batched=True)
    eval_tok = eval_ds.map(tok_fn, batched=True)

    # =======================
    # 7) Model
    # =======================
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(unique_labels),
        id2label=id2label,
        label2id=label2id,
    )

    # =======================
    # 8) Metrics
    # =======================
    import numpy as np
    from sklearn.metrics import f1_score, accuracy_score

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro")
        top3 = np.argsort(logits, axis=-1)[:, -3:]
        top3_acc = float(np.mean([labels[i] in top3[i] for i in range(len(labels))]))
        return {
            "accuracy": acc,
            "macro_f1": f1,
            "top3_acc": top3_acc,
        }

    # =======================
    # 9) Training args
    # =======================
    args = TrainingArguments(
        output_dir=OUT_DIR,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        logging_steps=20,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=SEED,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
    )

        # =======================
    # 10) Trainer
    # =======================
    trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    processing_class=tok,
    compute_metrics=compute_metrics,
    )

    # 开始训练
    train_result = trainer.train()

    import json
    import shutil
    from datetime import datetime

    # ===== 1) 实验目录（保留 checkpoint）=====
    EXP_DIR = "/root/autodl-tmp/experiments/cpc_classifier_v1"
    os.makedirs(EXP_DIR, exist_ok=True)

    # ===== 2) 发布目录（给本地调用，只放最终可用文件）=====
    RELEASE_DIR = "/root/autodl-tmp/experiments/cpc_classifier_release"
    os.makedirs(RELEASE_DIR, exist_ok=True)

    # 训练过程中最优 checkpoint
    best_ckpt = trainer.state.best_model_checkpoint

    # 如果没有 best checkpoint，就直接用当前 trainer model
    if best_ckpt is None:
        best_ckpt = EXP_DIR
        trainer.save_model(best_ckpt)

    print(f"Best checkpoint: {best_ckpt}")

    # ===== 3) 保存最终模型到发布目录 =====
    # 方式一：直接从 best checkpoint 复制模型文件
    # 方式二：重新 save_pretrained 到发布目录
    trainer.save_model(RELEASE_DIR)
    tok.save_pretrained(RELEASE_DIR)

    # ===== 4) 保存 label 映射 =====
    with open(os.path.join(RELEASE_DIR, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)

    with open(os.path.join(RELEASE_DIR, "id2label.json"), "w", encoding="utf-8") as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2)

    # ===== 5) 保存训练配置 =====
    train_config = {
        "model_name": MODEL_NAME,
        "local_jsonl": LOCAL_JSONL,
        "num_labels": len(unique_labels),
        "sample_labels": unique_labels[:50],
        "train_samples": len(train_tok),
        "eval_samples": len(eval_tok),
        "max_len": MAX_LEN,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "save_time": datetime.now().isoformat(),
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
    }

    with open(os.path.join(RELEASE_DIR, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(train_config, f, ensure_ascii=False, indent=2)

    # ===== 6) 保存训练指标 =====
    metrics = dict(train_result.metrics)
    metrics["best_model_checkpoint"] = trainer.state.best_model_checkpoint
    metrics["best_metric"] = trainer.state.best_metric

    with open(os.path.join(RELEASE_DIR, "train_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # ===== 7) 保存一个本地接入示例 =====
    inference_example = {
        "input_text": "An AI-assisted patent drafting system for e-commerce search ranking and product retrieval using dense retrieval, reranking, and online learning.",
        "output_format": [
            {"label": "G06", "probability": 0.73},
            {"label": "H04", "probability": 0.18},
            {"label": "G01", "probability": 0.04}
        ],
        "note": "本文件只是输出格式示例，不代表真实预测结果。"
    }

    with open(os.path.join(RELEASE_DIR, "inference_example.json"), "w", encoding="utf-8") as f:
        json.dump(inference_example, f, ensure_ascii=False, indent=2)

    print(f"Saved release model to: {RELEASE_DIR}")
    print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
    print(f"Num labels: {len(unique_labels)}")
    print("Sample labels:", unique_labels[:20])


if __name__ == "__main__":
    main()