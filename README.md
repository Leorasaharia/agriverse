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
## Frontend
### Unified Agricultural Advisor Interface

The Agriverse frontend is a Streamlit-based web application designed to make our multimodal AI agents accessible in a single, easy-to-use platform.

🌟 Frontend Capabilities

-Vision-based Crop Disease Detection
-Upload images of tomato or paddy leaves.
-Get instant classification results with detected disease name and confidence score.
-NER-based Entity Recognition (Hindi & Indic Languages)
-Extracts key agricultural entities such as crop names, pest names, and locations from user-provided text.
-Works with Hindi and other Indic languages for localized insights.

### Multilingual Agricultural Q&A 

-Chatbot powered by mT5-small for domain-specific question answering.
-Understands both Hindi and English queries.
-Provides context-aware, farmer-friendly recommendations.

### Knowledge Synthesis

Combines weather, crop cycles, pest control, soil health, and finance data into actionable guidance.


📱 User Experience

-Mobile & Low-Bandwidth Friendly — Optimized for rural usage.
-Multilingual UI for wider accessibility.
-Clean, intuitive design with clearly separated sections:
-Disease Detection
-NER Extraction
-Agri Chatbot

---

📸 Demo Platform Screenshots
Below are sample screenshots from the AgriVerse AI Gradio interface showing how our AI assists farmers with disease detection, entity recognition, and multilingual agricultural Q&A. <img width="1899" height="806" alt="synopsis1 (1)" src="https://github.com/user-attachments/assets/4a707735-fa65-4fe9-8b8c-355cd0e6d32d" />
<img width="1877" height="850" alt="synopsis1 (4)" src="https://github.com/user-attachments/assets/f395551a-74d9-4d18-9e55-de35ca6aef88" />
<img width="1891" height="470" alt="synopsis1 (3)" src="https://github.com/user-attachments/assets/3d5a70a1-e735-4bb4-b2d3-a089d13da1fd" />
<img width="1602" height="416" alt="synopsis1 (2)" src="https://github.com/user-attachments/assets/0b0241de-8643-4907-8460-6012d9c83cd8" />
<img width="970" height="888" alt="image" src="https://github.com/user-attachments/assets/12ad36f9-b40b-448c-b375-76bbf4af6092" />
<img width="1900" height="959" alt="image" src="https://github.com/user-attachments/assets/feb694e8-83a8-4f14-bcce-9a307a04aa61" />
<img width="907" height="369" alt="image" src="https://github.com/user-attachments/assets/ff80d2e3-6731-480f-926a-5fee987b91c9" />
<img width="911" height="711" alt="image" src="https://github.com/user-attachments/assets/a49e8161-d258-457d-b209-e45913b69131" />
<img width="921" height="479" alt="image" src="https://github.com/user-attachments/assets/6c6c3899-8914-450d-9a55-3054fbca932b" />
<img width="1139" height="769" alt="image" src="https://github.com/user-attachments/assets/05487428-24ec-4d72-8616-739da4b565fb" />

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

