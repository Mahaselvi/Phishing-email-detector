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

# ---------- Sample emails for demo ----------
SAMPLE_PHISHING_1 = """Subject: 𝗗𝗲𝗮𝗱𝗹𝗶𝗻𝗲 𝗔𝗽𝗽𝗿𝗼𝗮𝗰𝗵𝗶𝗻𝗴: 𝗨𝗚𝗖-𝗔𝗹𝗶𝗴𝗻𝗲𝗱 𝗜𝗻𝘁𝗲𝗿𝗻𝘀𝗵𝗶𝗽 𝗖𝗼𝗺𝗽𝗹𝗶𝗮𝗻𝗰𝗲 | 𝟮𝟬𝟮𝟲 𝗕𝗮𝘁𝗰𝗵 | 𝗦𝗔𝗦𝗧𝗥𝗔 𝗨𝗻𝗶𝘃𝗲𝗿𝘀𝗶𝘁𝘆
From: SRIRAMOJU RUSHI KUMAR SRIRAMOJU RUSHI KUMAR<21211a05v0@bvrit.ac.in>
undisclosed-recipients
129156098@sastra.ac.in
Dear Student,

To solve this growing employability gap, Unlox Academy has launched the JobBridge Professional Program under a Government-aligned Skill Development Initiative, making industry-focused training accessible to shortlisted students at a highly subsidized cost.

How to Apply:

Mandatory for students to complete the Form to register
Our HR Team will contact shortlisted candidates for the next steps.

Apply Now:- https://forms.gle/grST4719WiJKsyZ36

Join the Official WhatsApp Group: https://chat.whatsapp.com/GSRSRDn3ws7FkSk4RUsM6Z

This is why students across India are spending anywhere between ₹40,000 to ₹1,00,000+ on private professional programs just to improve their employability after college.

But the reality is — many of those programs offer only recorded classes, generic certificates, and very little real mentorship or placement support.

What Makes This Program Different?
Unlike traditional training platforms, this program focuses on complete career transformation through:

Industry Training
38+ Live Mentorship Sessions

Generative AI, Low-Code & Emerging Technologies

Flexible Evening Schedule

6-Month LMS Access + Learning Tablet

Structured Internship Experience
10+ Real-World Projects

1 Capstone Project

Industry-Simulated Work Environment

Mentor Evaluation & Performance Reports

Placement Preparation
AI-Based Mock Interviews

Resume & Portfolio Building

Weekly Career Bootcamps

Hiring & Job Opportunity Access

Verified Certifications
Upon successful completion, students receive:

Government-Aligned Certification with Verification ID

Co-Branded Internship Certificate

Digitally Verifiable Credentials

Aligned with:

NASSCOM

Skill India

Startup India

FutureSkills Prime Frameworks

Students who gain industry exposure early usually stand out during placements, while many others struggle later trying to catch up with market expectations.

The current intake has limited mentorship and internship slots, and registrations are being processed based on shortlisting.

Many students delay these opportunities assuming they will prepare “later” — but by the time placements begin, competition becomes much harder.

If you are serious about strengthening your resume, internship profile, and placement readiness, this may be one of the most valuable opportunities available during your academic journey.

Apply Now: https://forms.gle/grST4719WiJKsyZ36

Join the Official WhatsApp Group: https://chat.whatsapp.com/GSRSRDn3ws7FkSk4RUsM6Z

Regards,
Team Unlox Academy
Career & Industry Readiness Division


Engineering Sciences

BVRIT | SVECW | VIT | BVRITH

Medical Sciences

VDC | SVCP | VIPER | BVRICE
"""
 
SAMPLE_PHISHING_2 = """From: Gretchen Suggs <externalsep1@loanofficertool.com>
To: user2.2@gvc.ceas-challenge.cc
Tue, 05 Aug 2008 19:31:21 -0400
SpecialPricesPharmMoreinfo
WelcomeFastShippingCustomerSupport http://7iwfna.blu.livefilestore.com/y1pXdX3kwzhBa8xhXv8tdHbjHn7T...
"""
 
SAMPLE_LEGIT_1 = """Subject: TEJAS 2026 – AI Innovation Challenge / Hackathon & Codethon Poster and Event Details
From: Dr. V. S. Shankar Sriram .<sriram@it.sastra.edu>
To: tnjstudents@sastra.ac.in
Dear All,
Get ready to innovate, collaborate, and create impactful AI-driven solutions at TEJAS 2026, SASTRA’s exciting AI Innovation Challenge, Hackathon & Codethon!
Join forces with your team and turn your ideas into solutions for a smarter, more inclusive, and sustainable future.

TEJAS 2026 is planned to be conducted on 9th and 10th October 2026

Hackathon Main Theme: “AI Beyond Boundaries: Innovating for a Smarter, Inclusive and Sustainable Future”
■ Event Overview
• Event: TEJAS 2026 – AI Innovation Challenge / Hackathon & Codethon
• Scope: Intra-Institutional Competition at SASTRA
• Focus: Technology/AI-driven solutions aligned with selected UN Sustainable Development Goals (SDGs)
■ Eligibility
• Undergraduate: B.Tech – 2nd & 3rd Year students only, from all departments.
• Postgraduate: 1st Year M.Tech students.
• Postgraduate: 1st Year MCA students.
• Postgraduate: 1st Year M.Sc. Data Science students.
■ Team Size & Composition
• Each team must consist of exactly 5 members.
• All team members must belong to the same School within SASTRA (e.g., SoC, SEEE, SCBT, SoME, SoCE and M.Sc. Data Science).
• One member must be designated as the Team Leader.
• The Team Leader will be the primary contact person for registration and further communication.
• Students must form a team of 5 members before completing the registration.
• The Google Form must be filled and submitted only by the Team Leader on behalf of the complete team.

Good Luck!!

For further details contact chandramouli@sastra.edu
"""
 
SAMPLE_LEGIT_2 = """From: Prabakar T.N <prabakar@ece.sastra.edu>
Date: Sat, Jul 25, 2026 at 9:02 PM
Subject: Evening lab in SEEE - Starting
To: Naren PR <prnaren@scbt.sastra.ac.in>, Dr. K. Thenmozhi . <thenmozhik@ece.sastra.edu>, Dr. N. S. Manigandan . <manigandanns@eie.sastra.edu>


Dear Sir,
Want to share the following information to all students.
Like last semester, Evening lab [AI LAB in VV Block] is available till 7.30 PM for girls and 8.30 PM for boys like last semester. Please share the following with all the students. 
**************************
Dear Students,

From today onwards, the Evening Lab will be open during the following hours:

🕔 Girls: 5:15 PM – 7:30 PM
🕗 Boys: 5:15 PM – 8:30 PM

Please adhere to the following instructions:

1. Use the lab internet only for academic and research-related work. Do not use it for  social media, entertainment, or any other non-academic/unwanted browsing.
2. Enter your details correctly in the Lab Log Register, including both In Time and Out Time, every time you use the lab.
3. Do not take or move any lab equipment, components, or other objects without prior permission from the Lab In-charge or the concerned staff.
4. Maintain discipline, keep the lab clean, and use all laboratory resources responsibly and efficiently.

Let's make the best use of this opportunity to enhance your learning and research.

Make the best use of the same. All the best!
*****************************

With thanks & regards,
Dr T N Prabakar
Associate Professor / ECE
SEEE, SASTRA
Thanjavur 613401
prabakar@ece.sastra.edu
tnprabakar@gmail.com
+917010576716
"""
if "raw_email_input" not in st.session_state:
    st.session_state.raw_email_input = ""
 
def _set_sample(text):
    st.session_state.raw_email_input = text
 
st.write("**Try a sample email:**")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.button("Phishing Example 1", on_click=_set_sample, args=(SAMPLE_PHISHING_1,))
with col2:
    st.button("Phishing Example 2", on_click=_set_sample, args=(SAMPLE_PHISHING_2,))
with col3:
    st.button("Legitimate Example 1", on_click=_set_sample, args=(SAMPLE_LEGIT_1,))
with col4:
    st.button("Legitimate Example 2", on_click=_set_sample, args=(SAMPLE_LEGIT_2,))

# ---------- Input ----------
raw_email = st.text_area(
    "Paste the email here:",
    height=300,
    placeholder="From: someone@example.com\nTo: you@example.com\nSubject: ...\n\nBody text...",
    key="raw_email_input"
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
            if override_triggered:
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
        st.write("**Key Factors for the mail to be classified as** "+ ("**Legitimate:**" if prediction==0 else "**Phishing:**"))
        ordered_sentences = generate_explanations(top_features,prediction)
        for sentence in ordered_sentences:
            st.write(f"- {sentence}")

# ---------- Footer ----------
st.divider()
st.caption("This is a hackathon research prototype, not a certified security tool.")
