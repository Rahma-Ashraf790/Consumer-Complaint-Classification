import os
import gradio as gr
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import numpy as np  
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download("stopwords")
nltk.download("wordnet")

# Model & Path Settings 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "transformer_model")

LABELS = [
    "credit_card",
    "credit_reporting",
    "debt_collection",
    "mortgages_and_loans",
    "retail_banking"
]

print("Loading Model and Tokenizer from local path...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()  
print("Model loaded successfully!")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Brand Theming via Custom CSS
custom_css = """
* {
    box-sizing: border-box;
}

body, .gradio-container {
    background: #F0F4F8 !important; 
    color: #1E293B !important; 
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    max-width: 1450px !important; 
    margin: 0 auto !important;
    padding: 10px !important;
}

@keyframes softPulse {
    0% {
        transform: scale(1);
        box-shadow: 0 10px 30px rgba(53, 88, 114, 0.1);
    }
    50% {
        transform: scale(1.008);
        box-shadow: 0 15px 40px rgba(53, 88, 114, 0.18);
    }
    100% {
        transform: scale(1);
        box-shadow: 0 10px 30px rgba(53, 88, 114, 0.1);
    }
}

.unified-animated-header {
    background: linear-gradient(115deg, #FFFFFF 75%, #F0F4F8 100%) !important;
    border-radius: 20px !important;
    padding: 35px !important;
    text-align: center;
    max-width: 1050px !important; 
    margin: 25px auto !important;
    border: 2px solid #355872 !important;
    animation: softPulse 4s infinite ease-in-out; 
    transition: all 0.3s ease;
}

.unified-animated-header h1 {
    color: #355872 !important;
    font-size: 2.5em !important;
    font-weight: 800 !important;
    margin-bottom: 15px;
    margin-top: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px; 
}

/* Animated Border Header specifically for the second chart */
.animated-chart-header {
    border: none !important;
    background: #FFFFFF !important;
    border-radius: 12px !important;
    padding: 12px 20px !important;
    margin-bottom: 20px !important;
    display: inline-block;
    animation: softPulse 4s infinite ease-in-out;
}

.animated-chart-header h3 {
    color: #355872 !important;
    font-size: 1.25em !important;
    font-weight: bold !important;
    margin: 0 !important;
    display: flex;
    align-items: center;
    gap: 8px;
}

.project-description {
    color: #334155 !important;
    font-size: 1.08em;
    line-height: 1.65;
    max-width: 920px;
    margin: 0 auto;
    text-align: center;
}

.info-card {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important; 
    border-radius: 14px !important;
    padding: 22px 24px !important;
    text-align: center;
    box-shadow: 0 8px 20px rgba(53, 88, 114, 0.06) !important; 
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    height: 100%;
}

.info-card:hover {
    transform: translateY(-6px) !important;
    box-shadow: 0 15px 30px rgba(53, 88, 114, 0.15) !important;
    background: #FFFFFF !important; 
    border: 2px solid #355872 !important; 
}

.info-card-icon {
    font-size: 1.7em;
    margin-bottom: 8px;
}

.info-card-title {
    color: #355872 !important;
    font-weight: 700;
    font-size: 1.1em;
    margin-bottom: 6px;
}

.info-card-desc {
    color: #475569 !important;
    font-size: 0.92em;
    line-height: 1.5;
}

.second-row-margin {
    margin-top: 20px !important;
}

.toggle-workspace-btn {
    background: linear-gradient(135deg, #355872 0%, #253F53 100%) !important;
    color: #FFFFFF !important;
    font-weight: bold !important;
    font-size: 1.1em !important;
    border-radius: 12px !important;
    border: none !important;
    padding: 15px 35px !important;
    max-width: 320px !important;
    margin: 30px auto !important;
    display: block !important;
    box-shadow: 0 6px 20px rgba(53, 88, 114, 0.25) !important;
    transition: all 0.3s ease !important;
}

.toggle-workspace-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(53, 88, 114, 0.35) !important;
}

.main-large-workspace-box {
    background: #E2E8F044 !important; 
    border: 1px solid #CBD5E1 !important;
    border-radius: 16px !important;
    padding: 30px !important; 
    margin-top: 10px;
    margin-bottom: 20px !important;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.02) !important;
}

.content-white-card {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 14px !important;
    padding: 28px !important; 
    margin-bottom: 20px !important;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.04) !important;
}

.card-title {
    color: #355872 !important;
    font-size: 1.25em;
    font-weight: bold;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.progress-container {
    margin: 14px 0;
    padding: 12px 16px;
    background: #F8FAFC;
    border-radius: 8px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 2px 5px rgba(0,0,0,0.01) !important;
}

.progress-label {
    display: flex;
    justify-content: space-between;
    color: #334155;
    font-size: 0.95em;
    margin-bottom: 6px;
    font-weight: 600;
}

.progress-bar-bg {
    background: #E2E8F0;
    border-radius: 6px;
    height: 12px;
    overflow: hidden;
}

.progress-bar-fill {
    background: linear-gradient(90deg, #355872 0%, #253F53 100%);
    height: 100%;
    border-radius: 6px;
}

.blue-action-button {
    background: #355872 !important;
    color: #FFFFFF !important;
    font-weight: bold !important;
    border-radius: 8px !important;
    border: none !important;
    box-shadow: 0 4px 10px rgba(53, 88, 114, 0.2) !important;
    height: 45px;
    transition: all 0.2s ease;
}

.blue-action-button:hover {
    background: #253F53 !important;
    transform: translateY(-1px);
    box-shadow: 0 6px 15px rgba(53, 88, 114, 0.3) !important;
}

.gray-clear-button {
    background: #E2E8F0 !important;
    color: #475569 !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
    height: 45px;
    transition: all 0.2s ease;
}

.gray-clear-button:hover {
    background: #CBD5E1 !important;
}

textarea, .gr-textbox input {
    background: #F8FAFC !important;
    color: #1E293B !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
}

#routing-rec-box span {
    font-weight: 800 !important;
    color: #355872 !important;
}

#routing-rec-box textarea {
    font-weight: 400 !important; 
    color: #334155 !important;
    font-size: 1.05em !important;
    background: #F8FAFC !important;
}

.custom-system-footer {
    text-align: center !important;
    padding: 25px 10px 10px 10px !important;
    margin-top: 40px !important;
    border-top: 1px solid #E2E8F0 !important;
    color: #94A3B8 !important;
    font-size: 0.85em !important;
    line-height: 1.6;
}
"""

CATEGORY_NAMES = {
    "credit_reporting": "Credit Reporting",
    "debt_collection": "Debt Collection",
    "mortgages_and_loans": "Mortgages and Loans",
    "credit_card": "Credit Card",
    "retail_banking": "Retail Banking"
}

def preprocess_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [word for word in text.split() if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

def analyze_complaint(complaint_text):
    if not complaint_text or complaint_text.strip() == "":
        return (
            build_progress_bars({"credit_reporting": 0.0,"debt_collection": 0.0,"mortgages_and_loans": 0.0,"credit_card": 0.0,"retail_banking": 0.0}), 
            None, None, "Awaiting evaluation input...",
            build_card_html("None", 0, "Inactive")
        )

    inputs = tokenizer(
        complaint_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1).numpy()[0]

    categories = {
        LABELS[i]: float(probs[i])
        for i in range(len(LABELS))
    }

    predicted_class = LABELS[np.argmax(probs)]
    confidence = np.max(probs)
    predicted_name = CATEGORY_NAMES[predicted_class]

    sns.set_theme(style="whitegrid")
    
    # Primary distribution bar chart
    fig_bar, ax_bar = plt.subplots(figsize=(10, 3.5))
    cat_names = [CATEGORY_NAMES[c] for c in categories.keys()]
    vals = list(categories.values())
    custom_palette = ['#355872', '#476D8B', '#5A81A3', '#6E96BC', '#83ACD4']
    
    sns.barplot(x=vals, y=cat_names, palette=custom_palette, ax=ax_bar)
    ax_bar.set_xlim(0, 1)

    for i, val in enumerate(vals):
        ax_bar.text(
            val + 0.02,
            i,
            f"{val:.1%}",
            va="center",
            fontweight='bold',
            color='#1E293B'
        )
    plt.tight_layout()

    # NEW Donut Chart for Linguistic Metrics
    fig_density, ax_density = plt.subplots(figsize=(6, 4))
    raw_words = complaint_text.split()
    total_words = len(raw_words)
    total_chars = len(complaint_text)
    unique_words = len(set([w.lower().strip(string.punctuation) for w in raw_words]))
    
    metrics_labels = [f'Words\n({total_words})', f'Characters\n({total_chars})', f'Unique Vocab\n({unique_words})']
    metrics_vals = [total_words, total_chars, unique_words]
    
    # Visualizing as a Donut Chart 
    donut_colors = ['#355872', '#5A81A3', '#83ACD4']
    wedges, texts, autotexts = ax_density.pie(
        metrics_vals, 
        labels=metrics_labels, 
        colors=donut_colors, 
        autopct='%1.1f%%', 
        startangle=90, 
        pctdistance=0.75,
        textprops=dict(color="#1E293B", fontweight="bold")
    )
    
    # Draw central white circle to make it a donut chart
    centre_circle = plt.Circle((0,0), 0.50, fc='white')
    ax_density.add_artist(centre_circle)
    
    plt.setp(autotexts, size=9, weight="bold", color="white")
    ax_density.axis('equal')  
    plt.tight_layout()

    recommendation = f"Forward complaint to the {predicted_name} department."
    status_str = "High Priority" if confidence > 0.7 else "Normal Priority"
    cards_html = build_card_html(predicted_name, int(confidence * 100), status_str)

    return (
        build_progress_bars(categories),
        fig_bar,
        fig_density,
        recommendation,
        cards_html
    )

def build_card_html(dept, conf, status):
    return f"""
    <div style="display: flex; gap: 15px; justify-content: space-between; margin-bottom: 20px; flex-wrap: wrap;">
        <div class="info-card" style="flex: 1; min-width: 200px; border-top: 4px solid #355872 !important; box-shadow: 0 6px 15px rgba(0,0,0,0.04) !important;">
            <div class="info-card-icon" style="font-size: 1.4em;">🏢</div>
            <div class="info-card-title" style="font-size: 0.95em; color: #64748B !important;">Target Department</div>
            <div style="font-size: 1.25em; font-weight: 800; color: #355872; margin-top: 5px;">{dept}</div>
        </div>
        <div class="info-card" style="flex: 1; min-width: 200px; border-top: 4px solid #355872 !important; box-shadow: 0 6px 15px rgba(0,0,0,0.04) !important;">
            <div class="info-card-icon" style="font-size: 1.4em;">🎯</div>
            <div class="info-card-title" style="font-size: 0.95em; color: #64748B !important;">Confidence Score</div>
            <div style="font-size: 1.25em; font-weight: 800; color: #355872; margin-top: 5px;">{conf}%</div>
        </div>
        <div class="info-card" style="flex: 1; min-width: 200px; border-top: 4px solid #355872 !important; box-shadow: 0 6px 15px rgba(0,0,0,0.04) !important;">
            <div class="info-card-icon" style="font-size: 1.4em;">⚡</div>
            <div class="info-card-title" style="font-size: 0.95em; color: #64748B !important;">Routing Status</div>
            <div style="font-size: 1.25em; font-weight: 800; color: #355872; margin-top: 5px;">{status}</div>
        </div>
    </div>
    """

def build_progress_bars(categories):
    html = ""
    sorted_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)
    for name, value in sorted_cats:
        percent = int(value * 100)
        display_name = CATEGORY_NAMES.get(name, name.replace('_', ' ').title())
        html += f"""
        <div class="progress-container">
            <div class="progress-label">
                <span>{display_name}</span>
                <span style="color:#355872;">{percent}%</span>
            </div>
            <div class="progress-bar-bg">
                <div class="progress-bar-fill" style="width: {percent}%"></div>
            </div>
        </div>
        """
    return html

def toggle_workspace(is_visible):
    if is_visible:
        return gr.update(visible=False), gr.update(value="Show Diagnostics Workspace 📊"), False
    else:
        return gr.update(visible=True), gr.update(value="Hide Diagnostics Workspace ✖️"), True

def create_interface():
    with gr.Blocks(css=custom_css, theme=gr.themes.Base()) as demo:
        
        workspace_visible_state = gr.State(value=False)
        
        gr.HTML("""
        <div class="unified-animated-header">
            <h1><span>🏦</span> AI-Driven Consumer Complaint Management System</h1>
            <div class="project-description">
                An end-to-end intelligent optimization workflow designed to automatically parse, evaluate, and categorize incoming customer submissions into precise operational lines. This diagnostic hub calculates real-time multi-class score probabilities and risk severity factors to streamline back-office dispatch processes.
            </div>
        </div>
        """)
        
        with gr.Row():
            for icon, title, desc in [
                ("⚡", "Real-Time Processing", "Instantly analyzes unstructured text to deliver classification breakdowns under 2 seconds."),
                ("📊", "Confidence Metrics", "Provides probability distribution scores across multiple operational categories seamlessly."),
                ("🛡️", "Automated Triage", "Flags potential high-risk cases and suggests immediate internal routing actions.")
            ]:
                with gr.Column(scale=1, elem_classes=["info-card"]):
                    gr.HTML(f'<div class="info-card-icon">{icon}</div>')
                    gr.HTML(f'<div class="info-card-title">{title}</div>')
                    gr.HTML(f'<div class="info-card-desc">{desc}</div>')
        
        with gr.Row(elem_classes=["second-row-margin"]):
            for icon, title, desc in [
                ("📈", "Predictive Insights", "Extracts hidden patterns from consumer behaviors to predict routing trends accurately."),
                ("⚙", "Dynamic Workflow", "Adapts seamlessly to operational shifts and system load variations."),
                ("📁", "Centralized Logs", "Maintains structured output records for auditing and future model refinement.")
            ]:
                with gr.Column(scale=1, elem_classes=["info-card"]):
                    gr.HTML(f'<div class="info-card-icon">{icon}</div>')
                    gr.HTML(f'<div class="info-card-title">{title}</div>')
                    gr.HTML(f'<div class="info-card-desc">{desc}</div>')

        toggle_btn = gr.Button("Show Diagnostics Workspace 📊", elem_classes=["toggle-workspace-btn"])

        # Core Workspace Layout
        with gr.Column(elem_classes=["main-large-workspace-box"], visible=False) as workspace_box:
            
            # Input Control Panel Card
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["content-white-card"]):
                    gr.HTML('<div class="card-title">📝 Customer Complaint Input</div>')
                    
                    complaint_input = gr.Textbox(
                        label="Enter Complaint Text Here",
                        placeholder="Type customer submission details here to evaluate classification routing flow...",
                        lines=5
                    )
                    with gr.Row():
                        analyze_btn = gr.Button("Analyze Text", elem_classes=["blue-action-button"])
                        clear_btn = gr.Button("Clear", elem_classes=["gray-clear-button"])
            
            # Evaluation Output Card
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["content-white-card"]):
                    gr.HTML('<div class="card-title">🔬 Diagnostic Workspace & Results</div>')
                    
                    initial_categories = {"credit_reporting": 0.0, "debt_collection": 0.0, "mortgages_and_loans": 0.0, "credit_card": 0.0, "retail_banking": 0.0}
                    progress_html = gr.HTML(value=build_progress_bars(initial_categories))
                    
                    gr.HTML('<div style="margin-top:10px; font-weight:600; font-size:1.05em; color:#355872; margin-bottom:10px;">Executive Summary Cards</div>')
                    result_cards_box = gr.HTML(value=build_card_html("None", 0, "Waiting..."))
                    
                    recommendation_output = gr.Textbox(
                        label="AI Routing Recommendation",
                        value="Awaiting evaluation input...",
                        lines=2,
                        interactive=False,
                        elem_id="routing-rec-box"
                    )
            
            # Statistical Graphics Section
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["content-white-card"]):
                    gr.HTML('<div class="card-title">📈 Current Probability Vector Analysis</div>')
                    bar_chart = gr.Plot(label="", show_label=False)
                    
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["content-white-card"]):
                    gr.HTML("""
                    <div class="animated-chart-header">
                        <h3><span>📊</span> Linguistic Input Length & Density Metrics</h3>
                    </div>
                    """)
                    density_chart = gr.Plot(label="", show_label=False)
        
        gr.HTML("""
        <div class="custom-system-footer">
            © 2026 AI-Driven Consumer Complaint Management System • All Rights Reserved.<br>
            Powered by Advanced Deep Learning Classifiers Engine • Operational Dashboard Framework v3.4.0
        </div>
        """)
        
        toggle_btn.click(
            fn=toggle_workspace,
            inputs=[workspace_visible_state],
            outputs=[workspace_box, toggle_btn, workspace_visible_state]
        )
        
        analyze_btn.click(
            fn=analyze_complaint,
            inputs=[complaint_input],
            outputs=[progress_html, bar_chart, density_chart, recommendation_output, result_cards_box]
        )
        
        clear_btn.click(
            fn=lambda: ("", 
            build_progress_bars({"credit_reporting": 0.0, "debt_collection": 0.0, "mortgages_and_loans": 0.0, "credit_card": 0.0, "retail_banking": 0.0}),
            None, None, "Awaiting evaluation input...", build_card_html("None", 0, "Waiting...")),
            inputs=[],
            outputs=[complaint_input, progress_html, bar_chart, density_chart, recommendation_output, result_cards_box]
        )

    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="127.0.0.1", 
    server_port=7861,
     share=False
    )