"""
shap_explainer.py
Converts raw SHAP values into plain-language sentences, for structural
features AND TF-IDF word features alike.
"""

# Human-readable templates for structural features.
# {direction} gets replaced with "increased"/"decreased"
STRUCTURAL_TEMPLATES = {
    "num_urls": "The number of links in the email {direction} the phishing likelihood",
    "has_ip_url": "The email containing a link with a raw IP address {direction} the phishing likelihood",
    "has_at_in_url": "The presence of '@' inside a link {direction} the phishing likelihood",
    "sender_domain_mismatch": "The sender's domain not matching the links in the email {direction} the phishing likelihood",
    "sender_name_domain_mismatch": "The sender's display name not matching their email domain {direction} the phishing likelihood",
    "has_html": "The email being in HTML format {direction} the phishing likelihood",
    "has_form": "The email containing an embedded form {direction} the phishing likelihood",
    "has_iframe": "The email containing an embedded iframe {direction} the phishing likelihood",
    "urgent_keyword_count": "The use of urgency-related language {direction} the phishing likelihood",
}


def _strength_word(abs_value: float, max_abs_value: float) -> str:
    """Buckets magnitude into a strength word, relative to the strongest
    contributor in this specific prediction (so it's always meaningful,
    regardless of the model's absolute value scale)."""
    if max_abs_value == 0:
        return "slightly"
    ratio = abs_value / max_abs_value
    if ratio >= 0.66:
        return "strongly"
    elif ratio >= 0.33:
        return "moderately"
    else:
        return "slightly"


def explain_contribution(feature_name: str, shap_value: float, max_abs_value: float) -> str:
    """Turns one (feature_name, shap_value) pair into a plain-language sentence."""
    direction = "increased" if shap_value > 0 else "decreased"
    strength = _strength_word(abs(shap_value), max_abs_value)

    if feature_name in STRUCTURAL_TEMPLATES:
        sentence = STRUCTURAL_TEMPLATES[feature_name].format(direction=direction)
    elif feature_name.startswith("tfidf:"):
        word = feature_name.split("tfidf:", 1)[1].strip()
        sentence = f'The presence of the word "{word}" {direction} the phishing likelihood'
    else:
        sentence = f"The feature '{feature_name}' {direction} the phishing likelihood"

    return f"{sentence} ({strength})."


def generate_explanations(top_contributions: list) -> list:
    """
    top_contributions: list of (feature_name, shap_value) tuples, already
    sorted by abs(shap_value) descending (top-N slice).
    Returns a list of plain-language sentences, one per contribution.
    """
    if not top_contributions:
        return ["No strong individual signals were found — the prediction relies on a broad combination of weaker factors."]

    max_abs_value = max(abs(v) for _, v in top_contributions)
    return [explain_contribution(name, value, max_abs_value) for name, value in top_contributions]