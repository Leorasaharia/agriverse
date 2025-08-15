# streamlit.py
# AgriVerse AI: Train + Predict (Tomato/Paddy) + Hindi/English NER

import os
import sys
import subprocess
from typing import Dict, List

# -----------------------------
# Safe optional dependency install
# -----------------------------
def ensure(pkgs: List[str]):
    missing = []
    for p in pkgs:
        try:
            __import__(p.split("==")[0].split("[")[0])
        except Exception:
            missing.append(p)
    if missing:
        # Best-effort install; errors will surface later in imports if any
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)
        except Exception:
            pass

ensure([
    "streamlit",
    "torch",
    "torchvision",
    "transformers>=4.35.0",
    "datasets",
    "evaluate",
    "numpy",
    "Pillow",
    "seqeval",
])

import streamlit as st
import numpy as np
import torch
from PIL import Image
from datasets import load_dataset, Image as HFImage
import evaluate
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    pipeline,
)
import inspect
from torchvision import transforms

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="AgriVerse AI", layout="wide")
st.markdown(
    "<h1 style='text-align:center;'>🌱 AgriVerse AI</h1>"
    "<p style='text-align:center;'>Train & Predict crop diseases (Tomato/Paddy) + NER (Hindi/English)</p>",
    unsafe_allow_html=True,
)

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dynamic kwarg for transformers versions
EVAL_KWARG = "evaluation_strategy"
if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
    EVAL_KWARG = "eval_strategy"

# Default checkpoints & paths
VIT_BACKBONE = "google/vit-base-patch16-224-in21k"
TOMATO_OUTPUT_DIR = "tomato_vit"
PADDY_OUTPUT_DIR = "paddy_vit"

# Datasets (Hugging Face Hub)
# Tomato: curated public dataset used in your notebook
TOMATO_DATASET_ID = "wellCh4n/tomato-leaf-disease-image"  # images/labels
# Paddy: common HF dataset (falls back gracefully if unavailable)
PADDY_DATASET_ID = "keremberke/paddy-disease-classification"

# -----------------------------
# Small helpers
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_ner_pipe():
    """
    Hindi/English NER model (Naamapadam-style, multilingual).
    Using a well-known multilingual NER; you can swap to ai4bharat when needed.
    """
    # Try ai4bharat (if available), else fallback to XLM-RoBERTa multilingual NER
    model_candidates = [
        "ai4bharat/naamapadam-bert-base-multilingual-cased-ner",
        "Davlan/xlm-roberta-base-ner-hrl",
        "dslim/bert-base-NER",
    ]
    last_err = None
    for mid in model_candidates:
        try:
            return pipeline("token-classification", model=mid, aggregation_strategy="simple", device=0 if DEVICE=="cuda" else -1)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not load any NER model. Last error: {last_err}")

def vit_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

def exists_model_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.isdir(os.path.join(path, "model")) and os.path.isdir(os.path.join(path, "processor"))

def build_training_args(output_dir: str, epochs: int = 2, lr: float = 5e-5, bsz_train: int = 16, bsz_eval: int = 32):
    kwargs = {
        "output_dir": output_dir,
        "per_device_train_batch_size": bsz_train,
        "per_device_eval_batch_size": bsz_eval,
        "learning_rate": lr,
        "num_train_epochs": epochs,
        "save_strategy": "no",
        "fp16": (DEVICE == "cuda"),
        "report_to": "none",
        "logging_steps": 50,
        "remove_unused_columns": False,
        "dataloader_pin_memory": (DEVICE == "cuda"),
        EVAL_KWARG: "epoch",
    }
    return TrainingArguments(**kwargs)

def build_acc_metric():
    return evaluate.load("accuracy")

def make_log_container():
    return st.empty()

def write_log(log_box, msg: str):
    prev = st.session_state.get("LOGS", "")
    prev += f"{msg}\n"
    st.session_state["LOGS"] = prev
    log_box.code(prev)

def dataset_has_split(ds, split):
    try:
        _ = ds[split]
        return True
    except KeyError:
        return False

def prep_image_classification_dataset(dataset_id: str, img_col: str, label_col: str, processor):
    ds = load_dataset(dataset_id)
    if not isinstance(ds["train"].features[img_col], HFImage):
        ds = ds.cast_column(img_col, HFImage())

    labels = ds["train"].features[label_col].names
    id2label = {i: l for i, l in enumerate(labels)}
    label2id = {l: i for i, l in enumerate(labels)}

    def transform(batch):
        out = processor(images=batch[img_col], return_tensors="pt")
        out["labels"] = batch[label_col]
        return out

    # Keep training snappy by subselecting (you can increase for better accuracy)
    train_ds = ds["train"].shuffle(seed=SEED).select(
        range(min(3000, len(ds["train"])))
    ).with_transform(transform)

    val_split = "validation" if dataset_has_split(ds, "validation") else ("test" if dataset_has_split(ds, "test") else "train")
    eval_ds = ds[val_split].with_transform(transform)
    return train_ds, eval_ds, id2label, label2id, labels

def train_vit(dataset_id: str, output_dir: str, img_col: str = "image", label_col: str = "label"):
    st.toast(f"Starting training for {dataset_id} → {output_dir}", icon="🧪")
    log_box = make_log_container()
    write_log(log_box, f"Loading dataset: {dataset_id}")

    processor = AutoImageProcessor.from_pretrained(VIT_BACKBONE)
    train_ds, eval_ds, id2label, label2id, labels = prep_image_classification_dataset(dataset_id, img_col, label_col, processor)

    model = AutoModelForImageClassification.from_pretrained(
        VIT_BACKBONE,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    ).to(DEVICE)

    args = build_training_args(output_dir)
    metric = build_acc_metric()

    def compute_metrics(eval_pred):
        logits, y_true = eval_pred
        y_pred = np.argmax(logits, axis=-1)
        return metric.compute(predictions=y_pred, references=y_true)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )

    write_log(log_box, "🚀 Training...")
    trainer.train()
    write_log(log_box, "✅ Training finished. Evaluating...")
    results = trainer.evaluate()
    write_log(log_box, f"📊 Eval Results: {results}")

    # Save artifacts
    model_save = os.path.join(output_dir, "model")
    proc_save = os.path.join(output_dir, "processor")
    os.makedirs(model_save, exist_ok=True)
    os.makedirs(proc_save, exist_ok=True)
    model.save_pretrained(model_save)
    processor.save_pretrained(proc_save)
    write_log(log_box, f"💾 Saved to {output_dir}/(model|processor)")
    st.success(f"Training complete. Model saved under: {output_dir}", icon="✅")

# -----------------------------
# UI: Tabs
# -----------------------------
tab_train, tab_predict, tab_ner = st.tabs(["🧪 Train", "🔎 Predict", "📝 NER (Hindi/English)"])

# ========== TRAIN TAB ==========
with tab_train:
    st.subheader("Train Image Classification (ViT)")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Tomato Leaf Disease (ViT)**")
        if st.button("Train Tomato ViT", use_container_width=True):
            try:
                train_vit(TOMATO_DATASET_ID, TOMATO_OUTPUT_DIR, img_col="image", label_col="label")
            except Exception as e:
                st.error(f"Tomato training failed: {e}")

    with colB:
        st.markdown("**Paddy Disease (ViT)**")
        st.caption("Uses a public HF dataset; falls back gracefully if unavailable in your region.")
        if st.button("Train Paddy ViT", use_container_width=True):
            try:
                train_vit(PADDY_DATASET_ID, PADDY_OUTPUT_DIR, img_col="image", label_col="labels" if "labels" in load_dataset(PADDY_DATASET_ID)["train"].features else "label")
            except Exception as e:
                st.error(f"Paddy training failed: {e}")

    st.info(
        "Tip: Training here saves to `tomato_vit/` and `paddy_vit/`. "
        "The **Predict** tab will automatically use these folders."
    )

# ========== PREDICT TAB ==========
with tab_predict:
    st.subheader("Upload an image → Get prediction (Tomato/Paddy)")
    task = st.radio("Select model", ["Tomato (local ViT)", "Paddy (local ViT)"], horizontal=True)

    if task.startswith("Tomato"):
        model_dir = TOMATO_OUTPUT_DIR
    else:
        model_dir = PADDY_OUTPUT_DIR

    ready = exists_model_dir(model_dir)
    if not ready:
        st.warning(
            f"Model artifacts not found in `{model_dir}/model` and `{model_dir}/processor`.\n"
            f"Please train the model first in the **Train** tab.",
            icon="⚠️",
        )

    up = st.file_uploader("Upload JPG/PNG", type=["jpg", "jpeg", "png"])
    if up is not None:
        image = Image.open(up).convert("RGB")
        st.image(image, caption="Uploaded", use_column_width=True)

        if ready:
            try:
                processor = AutoImageProcessor.from_pretrained(os.path.join(model_dir, "processor"))
                model = AutoModelForImageClassification.from_pretrained(os.path.join(model_dir, "model")).to(DEVICE)
                model.eval()

                proc = processor(images=image, return_tensors="pt")
                pixel_values = proc["pixel_values"].to(DEVICE)
                with torch.no_grad():
                    logits = model(pixel_values).logits
                    probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()

                id2label: Dict[int, str] = model.config.id2label
                classes = [id2label[i] for i in range(len(id2label))]
                top_idx = int(np.argmax(probs))
                pred_label = classes[top_idx]
                pred_conf = float(probs[top_idx])

                st.success(f"Prediction: **{pred_label}**  |  Confidence: **{pred_conf:.2%}**", icon="✅")

                # Show a small table of top-k
                topk = min(5, len(classes))
                pairs = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)[:topk]
                st.table({"Class": [p[0] for p in pairs], "Confidence": [f"{p[1]:.2%}" for p in pairs]})

                # Simple domain nudge
                if "healthy" in pred_label.lower():
                    st.info("Looks healthy 👍. Keep monitoring; follow recommended irrigation & fertilization schedule.")
                else:
                    st.warning("Signs of disease detected. Consider isolating affected leaves and checking recommended treatments.")
            except Exception as e:
                st.error(f"Inference failed: {e}")
        else:
            st.stop()

# ========== NER TAB ==========
with tab_ner:
    st.subheader("Named Entity Recognition (Hindi / English)")
    st.caption("Extracts crop, location, pests, etc. from free text. Try mixing Hindi and English.")
    ner = None
    try:
        ner = get_ner_pipe()
    except Exception as e:
        st.error(f"Could not initialize NER model: {e}")

    sample = "मेरे धान में कीट लग गए हैं, रायपुर के पास खेत है। What pesticide should I use?"
    text = st.text_area("Enter text", value=sample, height=150)
    if st.button("Run NER", use_container_width=True) and ner is not None:
        try:
            outputs = ner(text)
            # Highlight
            highlighted = text
            # Sort spans in reverse so string indices remain valid while replacing
            spans = sorted(outputs, key=lambda x: (x["start"], x["end"]), reverse=True)
            for ent in spans:
                start, end = ent["start"], ent["end"]
                label = ent.get("entity_group", ent.get("entity", "ENT"))
                color = "#fde68a"  # light amber
                piece = highlighted[start:end]
                replacement = f"<span style='background:{color}; padding:2px 4px; border-radius:4px;'>{piece} <sub><b>{label}</b></sub></span>"
                highlighted = highlighted[:start] + replacement + highlighted[end:]
            st.markdown(highlighted, unsafe_allow_html=True)

            # Table
            st.markdown("**Entities**")
            st.table({
                "Text": [o["word"] for o in outputs],
                "Label": [o.get("entity_group", o.get("entity", "ENT")) for o in outputs],
                "Score": [f"{o['score']:.2f}" for o in outputs],
                "Start": [o["start"] for o in outputs],
                "End": [o["end"] for o in outputs],
            })
        except Exception as e:
            st.error(f"NER failed: {e}")

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    "<hr><p style='text-align:center; opacity:0.7;'>AgriVerse AI • "
    f"Device: <b>{DEVICE.upper()}</b> • Transformers eval key: <code>{EVAL_KWARG}</code></p>",
    unsafe_allow_html=True,
)
