import streamlit as st
import os
import random
import numpy as np
import torch
import transformers
import evaluate
from datasets import load_dataset, Image as HFImage
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer

# ===============================
# General setup
# ===============================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

st.set_page_config(page_title="AgriVerse AI Trainer", layout="wide")
st.title("🌱 AgriVerse AI Trainer")
st.write("Train and evaluate multiple AI models for agriculture datasets directly from your browser.")

# Sidebar controls
st.sidebar.header("Select Tasks to Run")
run_tomato_vit = st.sidebar.checkbox("🍅 Tomato Leaf Disease (ViT)", True)
run_paddy_vit = st.sidebar.checkbox("🌾 Paddy Disease (ViT)", False)
run_naamapadam_ner = st.sidebar.checkbox("📝 Naamapadam NER", False)
run_wikiann_ner = st.sidebar.checkbox("🌍 WikiAnn NER", False)
run_agroqa_qa = st.sidebar.checkbox("💬 AgroQA QA", False)
run_tomato_mbv3 = st.sidebar.checkbox("🍅 Tomato Leaf Disease (MobileNetV3)", False)

start_training = st.sidebar.button("🚀 Start Training")

# ===============================
# Utility
# ===============================
def log(msg):
    st.session_state.logs.append(msg)
    st.write(msg)

if "logs" not in st.session_state:
    st.session_state.logs = []

# ===============================
# Example: Tomato ViT training
# ===============================
def train_tomato_vit():
    log("Loading Tomato dataset...")
    ds = load_dataset("wellCh4n/tomato-leaf-disease-image")
    IMG_COL, LAB_COL = "image", "label"
    if not isinstance(ds["train"].features[IMG_COL], HFImage):
        ds = ds.cast_column(IMG_COL, HFImage())

    labels = ds["train"].features[LAB_COL].names
    id2label = {i: l for i, l in enumerate(labels)}
    label2id = {l: i for i, l in enumerate(labels)}

    ckpt = "google/vit-base-patch16-224-in21k"
    processor = AutoImageProcessor.from_pretrained(ckpt, use_fast=True)

    def transform(batch):
        out = processor(images=batch[IMG_COL], return_tensors="pt")
        out["labels"] = batch[LAB_COL]
        return out

    train_ds = ds["train"].shuffle(seed=SEED).select(range(min(3000, len(ds["train"])))).with_transform(transform)
    val_name = "validation" if "validation" in ds else ("test" if "test" in ds else "train")
    val_ds = ds[val_name].with_transform(transform)

    model = AutoModelForImageClassification.from_pretrained(
        ckpt,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    args = TrainingArguments(
        output_dir="tomato_vit",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=5e-5,
        num_train_epochs=2,
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        report_to="none",
        logging_steps=50,
        remove_unused_columns=False,
        dataloader_pin_memory=torch.cuda.is_available(),
        evaluation_strategy="epoch"
    )

    acc = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels_np = eval_pred
        preds = np.argmax(logits, axis=-1)
        return acc.compute(predictions=preds, references=labels_np)

    log("Starting training...")
    trainer = Trainer(model=model, args=args,
                      train_dataset=train_ds, eval_dataset=val_ds,
                      compute_metrics=compute_metrics)
    trainer.train()
    results = trainer.evaluate()
    log(f"Evaluation Results: {results}")

    model.save_pretrained("tomato_vit/model")
    processor.save_pretrained("tomato_vit/processor")
    log("✅ Model saved to tomato_vit/")

# ===============================
# Main run
# ===============================
if start_training:
    st.session_state.logs.clear()
    if run_tomato_vit:
        train_tomato_vit()
    if run_paddy_vit:
        log("Paddy ViT training not yet implemented.")
    if run_naamapadam_ner:
        log("Naamapadam NER training not yet implemented.")
    if run_wikiann_ner:
        log("WikiAnn NER training not yet implemented.")
    if run_agroqa_qa:
        log("AgroQA QA training not yet implemented.")
    if run_tomato_mbv3:
        log("Tomato MobileNetV3 training not yet implemented.")

# Show logs
st.subheader("📜 Training Logs")
for entry in st.session_state.logs:
    st.text(entry)
