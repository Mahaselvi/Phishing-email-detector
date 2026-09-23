import streamlit as st
import joblib
import numpy as np
import pandas as pd

from email_parser import parse_raw_email
from feature_extractor import build_feature_vector
from SHAP_Explainer import generate_explanations

# ---------- Load model artifacts (once, cached) ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    explainer = joblib.load("explainer.pkl")   # SHAP TreeExplainer, saved from the notebook
    structural_feature_names = [
        "num_urls", "has_ip_url", "has_at_in_url", "sender_domain_mismatch",
        "has_html", "has_form", "has_iframe",
        "urgent_keyword_count",
    ]
    tfidf_feature_names = [f"tfidf:{w}" for w in vectorizer.get_feature_names_out()]
    # must match the exact concat order used in build_feature_vector: tfidf first, then structural
    all_feature_names = tfidf_feature_names + structural_feature_names
    return model, vectorizer, explainer, all_feature_names

model, vectorizer, explainer, all_feature_names = load_artifacts()

# ---------- Page setup ----------
st.set_page_config(page_title="Phishing Email Detector", page_icon="🎣")
st.title("🎣 Phishing Email Detector")
st.write("Paste a full email (including headers, if available) to check if it's phishing.")

# ---------- Input ----------
raw_email = st.text_area(
    "Paste the email here:",
    height=300,
    placeholder="From: someone@example.com\nTo: you@example.com\nSubject: ...\n\nBody text..."
)

analyze_clicked = st.button("Analyze Email", type="primary")

# ---------- Analysis ----------
if analyze_clicked:
    if not raw_email.strip():
        st.warning("Please paste an email first.")
    else:
        with st.spinner("Analyzing..."):
            parsed = parse_raw_email(raw_email)
            feature_vector, structural_features = build_feature_vector(parsed, vectorizer)

            prediction = model.predict([feature_vector])[0]
            proba = model.predict_proba([feature_vector])[0]
            phishing_probability=proba[1]
            override_triggered=structural_features['has_ip_url']==1 or structural_features['sender_domain_mismatch']==1
            if override_triggered:
                prediction=1
                #phishing_probability=0.99 
            else:
                prediction=1 if phishing_probability>0.5 else 0
            confidence = phishing_probability* 100 if prediction==1 else (1-phishing_probability)*100

        # ---------- Result ----------
        st.divider()
        if prediction == 1:
            st.error(f"⚠️ **Phishing detected** ({confidence:.1f}% confidence)")
        else:
            st.success(f"✅ **Looks legitimate** ({confidence:.1f}% confidence)")

        # ---------- Parsed fields (transparency for the user/judges) ----------
        with st.expander("Parsed email fields"):
            st.write(f"**Sender:** {parsed['sender'] or '(not detected)'}")
            st.write(f"**Subject:** {parsed['subject'] or '(not detected)'}")
            st.write(f"**URLs found:** {len(parsed['urls'])}")
            if parsed['urls']:
                st.code("\n".join(parsed['urls']))

        # ---------- Explanation / indicators ----------
        st.subheader("Key indicators")
        flags = []
        if override_triggered:
            st.warning("⚠️ Flagged automatically: contains a raw IP-based link or sender/domain mismatch — a strong phishing indicator regardless of model confidence.")
        if structural_features["has_ip_url"]:
            flags.append("Contains a URL using a raw IP address instead of a domain")
        if structural_features["sender_domain_mismatch"]:
            flags.append("Sender domain does not match the domain(s) in the email's links")
        if structural_features["has_form"]:
            flags.append("Email contains an HTML form (often used to harvest credentials)")
        if structural_features["has_iframe"]:
            flags.append("Email contains an iframe")
        if structural_features["urgent_keyword_count"] > 0:
            flags.append(f"Contains {structural_features['urgent_keyword_count']} urgency-related keyword(s)")
        if structural_features["has_at_in_url"]:
            flags.append("Contains '@' inside a URL (used to disguise the real destination)")

        if flags:
            for f in flags:
                st.write(f"- {f}")
        else:
            st.write("No major structural red flags detected — prediction is mainly text-pattern based.")

        # ---------- SHAP-based explanation (model-driven, per-prediction) ----------
        st.subheader("Why the model made this decision")
        with st.spinner("Computing feature contributions..."):
            shap_values = explainer.shap_values(np.array([feature_vector]))
            # For binary classifiers, some SHAP explainers return a list [class0_vals, class1_vals]
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # contributions toward the "phishing" class
            shap_values = np.array(shap_values).flatten()

            contributions = list(zip(all_feature_names, shap_values))
            contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            top_contributions = contributions[:6]

        # Bar chart — visual, no raw numbers shown to the user
        chart_df = pd.DataFrame(top_contributions, columns=["feature", "shap_value"]).set_index("feature")
        st.bar_chart(chart_df)

        # Plain-language sentences generated from the same top contributions
        st.write("**In words:**")
        for sentence in generate_explanations(top_contributions):
            st.write(f"- {sentence}")

# ---------- Footer ----------
st.divider()
st.caption("This is a hackathon research prototype, not a certified security tool.")
