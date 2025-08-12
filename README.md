# 🌾 Agriverse — AI-Powered Agricultural Advisor

## Overview
Agriverse (Ava) is a **human-aligned AI-powered advisor** designed to answer all sorts of agriculture-related queries for **farmers, vendors, financiers, and policymakers**.  
It provides **real-time, contextual, and multilingual** support for questions like:

- "When should I irrigate wheat during winter?"
- "What seed variety suits this unpredictable weather?"
- "Will next week’s temperature drop kill my yield?"
- "Can I afford to wait for the market to improve?"
- "Where can I get affordable credit and government policy benefits?"

The goal is to **synthesize insights across domains** like weather, crop cycles, pest science, soil health, finance, and policy, while working under **low digital access constraints**.

---

## ✨ Key Features
- **Multimodal AI Agents** trained on:
  - Vision models for crop & disease detection
  - NER (Named Entity Recognition) for Indic languages
  - Multilingual QA using mT5 for agri-domain queries
- **Offline & Low-Bandwidth Friendly** deployment potential
- **Explainable & Reliable** responses to reduce wrong-decision risks
- **Grounded in factual datasets** to reduce LLM hallucinations

---

## 📂 Datasets Used
1. **Tomato Leaf Disease Classification**  
   - Source: [Hugging Face - Tomato Leaf](https://huggingface.co/datasets/emmarex/plantdisease)
   - Used for training a Vision Transformer to detect diseases in tomato leaves.

2. **Paddy Disease Classification**  
   - Source: [Hugging Face - Paddy Disease](https://huggingface.co/datasets/anthony2261/paddy-disease-classification)
   - Trained vision model for detecting paddy leaf diseases.

3. **Naamapadam NER (Hindi)**  
   - Source: [ai4bharat/naamapadam](https://huggingface.co/datasets/ai4bharat/naamapadam)
   - Used for Hindi NER to extract entities from agriculture-related text.

4. **WikiANN NER (Hindi)**  
   - Source: [wikiann](https://huggingface.co/datasets/wikiann)
   - Additional NER dataset for better generalization.

5. **AgroQA Multilingual Q&A**  
   - Source: Curated multilingual agri-domain Q&A dataset  
   - Trained using `google/mt5-small` for domain-specific question answering.

---

## 🛠 Tech Stack
- **Model Training:** Google Colab (GPU)
- **Frameworks:**  
  - Hugging Face `transformers`, `datasets`  
  - PyTorch  
  - `seqeval` for NER evaluation
- **Models:**  
  - Vision Transformer (ViT)  
  - MobileNetV3 (alternative lightweight vision model)  
  - XLM-RoBERTa (NER)  
  - mT5-small (multilingual QA)

---

📸 Demo Platform Screenshots
Below are sample screenshots from the AgriVerse AI Gradio interface showing how our AI assists farmers with disease detection, entity recognition, and multilingual agricultural Q&A.

---

## 🚀 How to Run
### 1. Clone the repo
```bash
git clone https://github.com/yourusername/agriverse.git
cd agriverse
````

### 2. Open Colab Notebook

[View in Colab](https://colab.research.google.com/drive/1eL7Q_RdV2hs71Ou4ErbEKlTzkrPzwNUY?usp=sharing)

### 3. Train Models

Toggle `RUN_*` flags in the notebook to train:

* Tomato Vision Model
* Paddy Vision Model
* Naamapadam Hindi NER
* WikiANN NER
* AgroQA mT5 QA Model

### 4. Test Inference

Example for QA:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tok = AutoTokenizer.from_pretrained("/content/agroqa_mt5/tokenizer")
model = AutoModelForSeq2SeqLM.from_pretrained("/content/agroqa_mt5/model").eval()

q = "question: When should I irrigate wheat during winter?"
gen = model.generate(**tok(q, return_tensors="pt"), max_new_tokens=64)
print(tok.decode(gen[0], skip_special_tokens=True))
```

Example for Hindi NER:

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
tok = AutoTokenizer.from_pretrained("/content/naamapadam_hi_ner/tokenizer")
model = AutoModelForTokenClassification.from_pretrained("/content/naamapadam_hi_ner/model").eval()
text = "राहुल ने पटना में किसान मेले का उद्घाटन किया।"
enc  = tok(text, return_tensors="pt", truncation=True)
out = model(**enc).logits.argmax(-1).squeeze().tolist()
tags = [model.config.id2label[i] for i in out]
print(list(zip(tok.convert_ids_to_tokens(enc["input_ids"].squeeze()), tags)))
```

---

## 📊 Current Limitations

* Multilingual QA model still needs better fine-tuning for **cross-domain synthesis** (weather + finance + policy).
* Requires additional domain-specific datasets for deeper policy/finance integration.

---

## 📅 Next Steps

* Merge all domain-specific datasets for **joint training**.
* Optimize for **edge devices** with quantized models.
* Integrate **speech-to-text** and **text-to-speech** for offline rural deployment.
* Expand **regional language support**.

---

## 🤝 Team

**Team Name:** Hackers
**Members:**

* Leora Saharia 
* Prayash Sinha 

---

