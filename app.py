import streamlit as st
import joblib
import numpy as np
import pandas as pd

from email_parser import parse_raw_email
from feature_extractor import build_feature_vector
from SHAP_Explainer import generate_explanations

# ---------- Load model artifacts (once, cached) ----------
@st.cache_resource(show_spinner="Phishing Email Detector loading...")
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
            has_ip=structural_features['has_ip_url']==1
            domain_mismatch=structural_features['sender_domain_mismatch']==1
            override_triggered=has_ip or domain_mismatch
            if domain_mismatch:
                prediction=1
                #phishing_probability=0.99 
            else:
                prediction=1 if phishing_probability>0.5 else 0
            confidence = phishing_probability* 100 if prediction==1 else (1-phishing_probability)*100

        # ---------- Result ----------
        st.divider()
        if prediction == 1:
            st.error(f"⚠️ **Phishing detected**")# ({confidence:.1f}% confidence)")
        else:
            st.success(f"✅ **Looks legitimate**")# ({confidence:.1f}% confidence)")

        # ---------- Parsed fields----------
        with st.expander("Parsed email fields"):
            st.write(f"**Sender:** {parsed['sender'] or '(not detected)'}")
            st.write(f"**Subject:** {parsed['subject'] or '(not detected)'}")
            st.write(f"**URLs found:** {len(parsed['urls'])}")
            if parsed['urls']:
                st.code("\n".join(parsed['urls']))

        # ---------- Explanation / indicators ----------
        st.subheader("Key indicators")
        flags = []
        if has_ip:
            st.warning("⚠️ **Critical Security Alert:** This email contains a URL using a raw IP address instead of a domain name (a classic phishing indicator).")
        if domain_mismatch:
            st.warning("⚠️ **Critical Security Alert:** The sender's domain does not match the domain found in the email's links (possible domain spoofing).")
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
        elif not override_triggered:
            st.write("No major structural red flags detected — prediction is mainly text-pattern based.")

        # ---------- SHAP-based explanation ----------
        st.subheader("Why the model made this decision")
        st.caption("📊 Explanation derived from SHAP (SHapley Additive exPlanations) values")
        with st.spinner("Computing feature contributions..."):
            shap_values = explainer.shap_values(np.array([feature_vector]))
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  
            shap_values = np.array(shap_values).flatten()

            contributions = list(zip(all_feature_names, shap_values))
            
            positives = [c for c in contributions if c[1] >= 0]
            negatives = [c for c in contributions if c[1] < 0]
            
            positives.sort(key=lambda x: abs(x[1]), reverse=True)
            negatives.sort(key=lambda x: abs(x[1]), reverse=True)

            if prediction == 1:
                # Phishing: Only top 6 risk-increasing features
                top_features = positives[:6]
                chart_data = positives[:6]
            else:
                # Legitimate: Only top 6 safety-supporting features
                top_features = negatives[:6]
                # Use absolute values for the chart so bars render cleanly above the axis
                chart_data = [(feat, abs(val)) for feat, val in negatives[:6]]

        # Render bar chart matching the context
        if chart_data:
            chart_df = pd.DataFrame(chart_data, columns=["feature", "shap_value"]).set_index("feature")
            st.bar_chart(chart_df)
        else:
            st.info("No significant contributing features found.")

        # Plain-language sentences using only the filtered top features
        st.write("**In words:**")
        ordered_sentences = generate_explanations(top_features)
        for sentence in ordered_sentences:
            st.write(f"- {sentence}")

# ---------- Footer ----------
st.divider()
st.caption("This is a hackathon research prototype, not a certified security tool.")
