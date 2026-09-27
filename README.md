 # 🎣 Phishing Email Detector

An AI system that classifies emails as **phishing** or **legitimate** using a combination of text signals (TF-IDF word patterns) and structural signals (links, sender domain behavior, HTML content) — and explains *why* it made each prediction using SHAP (SHapley Additive exPlanations).

Built as a college project by **Hema Sarika**, **Sorna Raaga Priya**, and **Mahaselvi**.

---

## 🧠 What it does

Paste any email (with headers, if available) into the app, and it will:

1. **Parse** the raw email into structured fields — sender, subject, body, and any URLs.
2. **Extract features** from those fields — both text-based (TF-IDF) and structural (e.g. does a link use a raw IP address, does the sender's display name not match their domain, is there urgency-driven language like "verify" or "act now").
3. **Predict** whether the email is phishing or legitimate using a trained classification model.
4. **Explain** the decision in plain English, using SHAP values to show exactly which words or structural signals pushed the prediction one way or the other — plus a bar chart of the top contributing features.

---

## 📊 Dataset

The model is trained on the **Nazario_5.csv** dataset, sourced from Kaggle:
[Phishing Email Dataset (Nazario-5 and Trec07)](https://www.kaggle.com/datasets/rohansood98/phishing-email-dataset-nazario-5-and-trec07)

It contains labeled phishing and legitimate emails with fields for sender, subject, body, URLs, and a binary label.

---

## 🏗️ How the project is structured

| File | Purpose |
|---|---|
| `phishing_email_notebook.ipynb` | The full training notebook — data loading, preprocessing, feature extraction, TF-IDF vectorization, model training (Random Forest / XGBoost), and evaluation. |
| `email_parser.py` | Parses a raw pasted email (or a dataset row) into structured fields: sender, subject, body, and URLs. |
| `feature_extractor.py` | Converts parsed email fields into structural features (URL count, IP-based links, HTML/form/iframe presence, sender-domain mismatches, urgency keywords) and combines them with TF-IDF text features into one feature vector. |
| `SHAP_Explainer.py` | Converts raw SHAP values into plain-language sentences explaining what pushed a prediction toward or away from "phishing." |
| `app.py` | The Streamlit web app — the user-facing interface where someone pastes an email and gets a prediction with an explanation. |
| `model.pkl` | The trained classification model, saved from the notebook. |
| `vectorizer.pkl` | The fitted TF-IDF vectorizer, saved from training (must stay paired with `model.pkl`, since both were trained together). |
| `explainer.pkl` | The saved SHAP explainer, used to compute feature contributions at prediction time. |
| `Nazario_5.csv` | The labeled email dataset used for training. |
| `Email-Phishing_and_Legitimate_count.png` | A chart showing the class balance (phishing vs. legitimate) in the dataset. |
| `confusion_matrix.png` | The model's confusion matrix on the test set, from evaluation. |
| `requirements.txt` | Python packages required to run the app. |

---

## ⚙️ How it works under the hood

- **Training-time and live-prediction feature logic is identical.** Whether an email comes from the training CSV or is pasted live by a user, it's converted into the same parsed-field format and run through the same `feature_extractor.py` — this consistency is what makes the trained model valid at prediction time.
- **Structural features** flag classic phishing patterns: links using raw IP addresses, `@` symbols hidden inside URLs, sender display names that don't match their actual domain, embedded HTML forms/iframes, and urgency-driven keywords.
- **Text features** come from a TF-IDF vectorizer fit on the subject + body text, capturing which words are unusually characteristic of phishing vs. legitimate emails.
- **Explainability** is handled by SHAP: for each prediction, the app computes which individual features (structural or word-based) contributed most, and `SHAP_Explainer.py` turns those into readable sentences like *"The email containing a link with a raw IP address increased the phishing likelihood (strongly)."*

---

## 🚀 Running the app locally

### 1. Clone the repository
```bash
git clone https://github.com/Mahaselvi/Phishing-email-detector.git
cd Phishing-email-detector
```

### 2. (Optional) Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

This will open the app in your browser (usually at `http://localhost:8501`).

> **Note:** You don't need to retrain anything or run the notebook to use the app — `model.pkl`, `vectorizer.pkl`, and `explainer.pkl` are already included in the repo, and `app.py` loads them directly. It also imports `email_parser.py`, `feature_extractor.py`, and `SHAP_Explainer.py` directly, so as long as all files stay in the same folder (as they are when cloned), everything works out of the box.

### 5. Try it out
The app includes sample phishing and legitimate emails you can load with one click, or you can paste your own email (including headers, if you have them) to test it.

---

## 🔬 Retraining the model (optional)

If you want to retrain the model on updated data, open `phishing_email_notebook.ipynb` and run it top to bottom. It will regenerate `model.pkl`, `vectorizer.pkl`, and `explainer.pkl` — just make sure to overwrite the existing ones in the project root so `app.py` picks up the new versions.

---

## ⚠️ Disclaimer

This is a college/hackathon research prototype, not a certified security tool. Predictions should not be relied upon as the sole basis for real-world security decisions.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
