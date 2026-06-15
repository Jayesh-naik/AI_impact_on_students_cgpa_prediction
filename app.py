import sys
import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# ==============================================================================
# 1. SCOPE & VERSION COMPATIBILITY SHIMS
# ==============================================================================
# The pickled model was saved with scikit-learn 1.6.1, which expects older internal
# module structures. Here, we patch the environment to ensure seamless loading.

# Shim for ColumnTransformer remainder cols list representation
import sklearn.compose._column_transformer as ct
if not hasattr(ct, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass
    ct._RemainderColsList = _RemainderColsList

# Shim for CyHalfSquaredError loss helper
import sklearn._loss._loss as ll
sys.modules['_loss'] = ll

# ==============================================================================
# 2. PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="Academic & GenAI Predictor",
    page_icon="🎓",
    layout="wide"
)

# Custom Styling (Premium Glassmorphism & Sleek Dark theme)
st.markdown("""
<style>
    /* Main container background overrides */
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #1e1b4b 100%);
        color: #f1f5f9;
    }
    
    /* Sleek card containers */
    .premium-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .premium-card:hover {
        transform: translateY(-2px);
        border-color: rgba(129, 140, 248, 0.3);
    }
    
    /* Text styling */
    .header-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
        text-shadow: 0 4px 12px rgba(99, 102, 241, 0.15);
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 30px;
    }
    
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 15px;
        border-bottom: 2px solid rgba(99, 102, 241, 0.2);
        padding-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Prediction block styling */
    .prediction-container {
        background: radial-gradient(circle at top left, rgba(99, 102, 241, 0.15) 0%, rgba(167, 139, 250, 0.05) 100%);
        border: 1.5px dashed rgba(129, 140, 248, 0.4);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 12px 40px 0 rgba(99, 102, 241, 0.15);
    }
    
    .prediction-value {
        font-size: 4.8rem;
        font-weight: 900;
        background: linear-gradient(135deg, #60a5fa 0%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 15px 0;
        letter-spacing: -2px;
        filter: drop-shadow(0px 2px 10px rgba(52, 211, 153, 0.2));
    }
    
    .delta-indicator {
        font-size: 0.95rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
        display: inline-block;
    }
    
    .delta-positive {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.3);
    }
    
    .delta-negative {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(248, 113, 113, 0.3);
    }
    
    /* Comparison progress bar labels */
    .compare-label {
        font-size: 0.85rem;
        color: #94a3b8;
        display: flex;
        justify-content: space-between;
        margin-bottom: 4px;
        margin-top: 10px;
    }
    
    /* Info badge styling */
    .info-badge {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 0.9rem;
        color: #cbd5e1;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. MODEL LOADING
# ==============================================================================
@st.cache_resource
def load_gpa_model():
    model_path = os.path.join(os.path.dirname(__file__), "gpa_predictor_pipeline_new.pkl")
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Failed to load the model file at {model_path}: {e}")
        return None

pipeline = load_gpa_model()

# ==============================================================================
# 4. APP HEADER & LAYOUT
# ==============================================================================
st.markdown('<div class="header-title">🎓 Academic & GenAI Insights Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="header-subtitle">Analyze study habits, generative AI adoption, and mental health factors to predict GPA.</div>', unsafe_allow_html=True)

if pipeline is not None:
    # We lay out inputs on the left (spanning 2 columns) and predictions on the right (1 column)
    col_input_left, col_input_right, col_pred = st.columns([1.1, 1.1, 1.3], gap="large")
    
    with col_input_left:
        # Academic Profile Card
        st.markdown("""
        <div class="premium-card">
            <div class="section-title">📚 Academic Profile</div>
        </div>
        """, unsafe_allow_html=True)
        
        pre_gpa = st.slider(
            "Pre-Semester GPA",
            min_value=0.00,
            max_value=4.00,
            value=3.15,
            step=0.01,
            help="Your cumulative GPA from the previous semester."
        )
        
        study_hours = st.slider(
            "Traditional Study Hours (Weekly)",
            min_value=0.0,
            max_value=50.0,
            value=11.2,
            step=0.5,
            help="Weekly hours spent studying using books, lectures, and non-AI methods."
        )
        
        retention_score = st.slider(
            "Skill Retention Score (%)",
            min_value=0.0,
            max_value=100.0,
            value=75.8,
            step=0.5,
            help="Your average score on concept retention evaluations."
        )
        
        exam_anxiety = st.slider(
            "Anxiety Level During Exams",
            min_value=1,
            max_value=10,
            value=4,
            step=1,
            help="Self-reported stress level during exams (1: Calm, 10: High Anxiety)."
        )
        
        # Contextual & Well-being Card
        st.markdown("""
        <div class="premium-card" style="margin-top: 25px;">
            <div class="section-title">🏢 Institutional Context</div>
        </div>
        """, unsafe_allow_html=True)
        
        policy_label_map = {
            "Actively_Encouraged": "Actively Encouraged",
            "Allowed_With_Citation": "Allowed with Citation",
            "Strict_Ban": "Strict Ban"
        }
        
        policy = st.selectbox(
            "Institutional GenAI Policy",
            options=list(policy_label_map.keys()),
            format_func=lambda x: policy_label_map[x],
            index=1,
            help="The academic policy regarding GenAI usage in your institution."
        )

    with col_input_right:
        # GenAI Usage Profile Card
        st.markdown("""
        <div class="premium-card">
            <div class="section-title">⚡ GenAI Integration</div>
        </div>
        """, unsafe_allow_html=True)
        
        genai_hours = st.slider(
            "Weekly GenAI Usage (Hours)",
            min_value=0.0,
            max_value=50.0,
            value=8.4,
            step=0.5,
            help="Hours spent using tools like ChatGPT, Claude, Gemini, etc. for academic tasks."
        )
        
        tool_div = st.slider(
            "Tool Diversity (Count)",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            help="Number of distinct AI tools regularly used in your academic workflow."
        )
        
        prompt_skill = st.select_slider(
            "Prompt Engineering Skill",
            options=["Beginner", "Intermediate", "Advanced"],
            value="Intermediate",
            help="Self-assessed capability in structuring prompts effectively."
        )
        
        paid_sub = st.checkbox(
            "Paid Subscription Holder",
            value=False,
            help="Do you subscribe to paid versions of AI tools (e.g. ChatGPT Plus, Claude Pro)?"
        )
        
        # AI Dependency Card
        st.markdown("""
        <div class="premium-card" style="margin-top: 22px;">
            <div class="section-title">🧠 AI Dependency</div>
        </div>
        """, unsafe_allow_html=True)
        
        ai_dependency = st.slider(
            "Perceived AI Dependency",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            help="Self-reported assessment of how dependent you are on AI to finish tasks (1: Independent, 10: Highly Dependent)."
        )

    with col_pred:
        # Create input DataFrame matching feature_names_in_ exactly
        input_data = {
            'Pre_Semester_GPA': [pre_gpa],
            'Weekly_GenAI_Hours': [genai_hours],
            'Prompt_Engineering_Skill': [prompt_skill],
            'Tool_Diversity': [tool_div],
            'Paid_Subscription': [paid_sub],
            'Traditional_Study_Hours': [study_hours],
            'Perceived_AI_Dependency': [ai_dependency],
            'Institutional_Policy': [policy],
            'Anxiety_Level_During_Exams': [exam_anxiety],
            'Skill_Retention_Score': [retention_score]
        }
        input_df = pd.DataFrame(input_data)
        
        # Predict Semester GPA
        predicted_gpa = pipeline.predict(input_df)[0]
        
        # Bound predicted GPA to practical limits [0.00, 4.00]
        predicted_gpa = float(np.clip(predicted_gpa, 0.00, 4.00))
        
        # Calculate Delta against Pre_Semester_GPA
        delta = predicted_gpa - pre_gpa
        
        # Render Prediction Container
        st.markdown('<div class="premium-card prediction-container">', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 1.1rem; text-transform: uppercase; letter-spacing: 1.5px; color: #a78bfa; font-weight: 700;">Predicted Semester GPA</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="prediction-value">{predicted_gpa:.3f}</div>', unsafe_allow_html=True)
        
        # Render beautiful Delta Badge
        if delta >= 0:
            st.markdown(f'<div class="delta-indicator delta-positive">📈 GPA Increase: +{delta:.3f}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="delta-indicator delta-negative">📉 GPA Decrease: {delta:.3f}</div>', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Render Insights & Comparison Visuals
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Key Comparisons</div>', unsafe_allow_html=True)
        
        # Compare GenAI vs Traditional Study Hours
        total_study_engagement = genai_hours + study_hours
        genai_ratio = (genai_hours / total_study_engagement * 100) if total_study_engagement > 0 else 0.0
        
        st.markdown(f"""
        <div class="compare-label">
            <span>GenAI Integration Ratio</span>
            <span>{genai_ratio:.1f}% of Study Time</span>
        </div>
        """, unsafe_allow_html=True)
        st.progress(min(max(genai_ratio / 100.0, 0.0), 1.0))
        
        # Compare Study Hours to Dataset Mean (11.2h)
        study_hours_rel = study_hours / 11.2064
        st.markdown(f"""
        <div class="compare-label">
            <span>Traditional Study (vs. Avg Student)</span>
            <span>{study_hours:.1f}h / 11.2h</span>
        </div>
        """, unsafe_allow_html=True)
        st.progress(min(max(study_hours_rel / 2.0, 0.0), 1.0)) # capped at 2x mean for progress bar
        
        # Compare GenAI Hours to Dataset Mean (8.4h)
        genai_hours_rel = genai_hours / 8.4380
        st.markdown(f"""
        <div class="compare-label">
            <span>GenAI Usage (vs. Avg Student)</span>
            <span>{genai_hours:.1f}h / 8.4h</span>
        </div>
        """, unsafe_allow_html=True)
        st.progress(min(max(genai_hours_rel / 2.0, 0.0), 1.0)) # capped at 2x mean for progress bar
        
        # Brief Contextual Advice Card
        st.markdown('<div class="info-badge">', unsafe_allow_html=True)
        if ai_dependency > 6 and genai_hours > 15.0:
            st.markdown("⚠️ **High AI Dependency Detected**: While GenAI tools accelerate study workflows, relying heavily on them (AI Dependency > 6) might affect long-term retention and final exam performance.")
        elif exam_anxiety > 7:
            st.markdown("🧘 **Exam Anxiety Warning**: High self-reported exam anxiety can significantly pull down performance regardless of study hours. Consider mindfulness exercises or testing accommodations.")
        else:
            st.markdown("✨ **Balanced Profile**: Your study habits, tool diversity, and AI dependency are within a healthy, productive range. Continue leveraging AI as a supportive study partner.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.error("Error: Could not render prediction dashboard. Verify the model file location.")