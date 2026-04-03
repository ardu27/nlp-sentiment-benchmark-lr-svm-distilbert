import streamlit as st
import joblib
import glob
import os
import eli5
import streamlit.components.v1 as components
import config
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.preprocessing import TextCleaner

st.set_page_config(page_title="AI Sentiment Analysis")
st.title("Sentiment Analysis System")
st.write("This application classifies reviews as Positive or Negative using multiple models.")

@st.cache_resource
def load_traditional_model(directory):
    if not os.path.exists(directory):
        return None, None
        
    list_of_files = glob.glob(os.path.join(directory, "*.pkl"))
    if not list_of_files:
        return None, None
        
    latest_file = max(list_of_files, key=os.path.getctime)
    model = joblib.load(latest_file)
    return model, latest_file

@st.cache_resource
def load_transformer():
    transformer_path = config.TRANSFORMER_SAVE_PATH
    if not os.path.exists(transformer_path):
        return None, None, None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(transformer_path)
    model_bert = AutoModelForSequenceClassification.from_pretrained(transformer_path)
    model_bert.to(device)
    model_bert.eval()
    return tokenizer, model_bert, device

model_lr, path_lr = load_traditional_model(config.MODELS_DIR_LR)
model_svm, path_svm = load_traditional_model(config.MODELS_DIR_SVM)
tokenizer, model_bert, device = load_transformer()
cleaner = TextCleaner()

tab_lr, tab_svm, tab_bert = st.tabs(["LR Model", "SVM Model", "DistilBERT"])

with tab_lr:
    if model_lr is None:
        st.warning("LR model not found. Run main.py first.")
    else:
        st.success(f"Using LR model: {os.path.basename(path_lr)}")
        user_input_lr = st.text_area("Enter a review here (in English):", height=150, key="lr_text")
        if st.button("Analyze Text with LR"):
            if user_input_lr.strip() == "":
                st.warning("Please enter a valid text.")
            else:
                cleaned_text = cleaner.full_clean([user_input_lr])[0]
                prediction = model_lr.predict([cleaned_text])[0]

                st.subheader("Result:")
                if prediction == 1:
                    st.success("That text is POSITIVE!")
                else:
                    st.error("That text is NEGATIVE!")

                st.subheader("Model Explanation (Explainable AI)")
                try:
                    explanation = eli5.explain_prediction(
                        model_lr.named_steps['lr'], 
                        cleaned_text, 
                        vec=model_lr.named_steps['tfidf']
                    )
                    html_expl = eli5.formatters.html.format_as_html(explanation)
                    responsive_css = "<style> .eli5-explanation, table.eli5-weights, table.eli5-weights th, table.eli5-weights td { font-family: 'Source Sans Pro', sans-serif !important; } @media (prefers-color-scheme: light) { .eli5-explanation, table.eli5-weights th { color: #212529 !important; } } @media (prefers-color-scheme: dark) { .eli5-explanation, table.eli5-weights th { color: #fafafa !important; } } table.eli5-weights td, .eli5-bg { color: #000000 !important; } </style>"
                    html_expl = responsive_css + html_expl
                    components.html(html_expl, height=450, scrolling=True)
                except KeyError:
                    st.warning("Model explanation is unavailable.")

with tab_svm:
    if model_svm is None:
        st.warning("SVM model not found. Run main.py first.")
    else:
        st.success(f"Using SVM model: {os.path.basename(path_svm)}")
        user_input_svm = st.text_area("Enter a review here (in English):", height=150, key="svm_text")
        if st.button("Analyze Text with SVM"):
            if user_input_svm.strip() == "":
                st.warning("Please enter a valid text.")
            else:
                cleaned_text = cleaner.full_clean([user_input_svm])[0]
                prediction = model_svm.predict([cleaned_text])[0]
                
                decision = model_svm.decision_function([cleaned_text])[0]
                prob_pos = 1 / (1 + np.exp(-decision))
                prob_neg = 1 - prob_pos
                confidence = prob_pos if prediction == 1 else prob_neg

                st.subheader("Result:")
                if prediction == 1:
                    st.success(f"That text is POSITIVE! (Confidence: {confidence:.2%})")
                else:
                    st.error(f"That text is NEGATIVE! (Confidence: {confidence:.2%})")

                st.subheader("Model Explanation (Explainable AI)")
                try:
                    explanation = eli5.explain_prediction(
                        model_svm.named_steps['svm'], 
                        cleaned_text, 
                        vec=model_svm.named_steps['tfidf']
                    )
                    html_expl = eli5.formatters.html.format_as_html(explanation)
                    responsive_css = "<style> .eli5-explanation, table.eli5-weights, table.eli5-weights th, table.eli5-weights td { font-family: 'Source Sans Pro', sans-serif !important; } @media (prefers-color-scheme: light) { .eli5-explanation, table.eli5-weights th { color: #212529 !important; } } @media (prefers-color-scheme: dark) { .eli5-explanation, table.eli5-weights th { color: #fafafa !important; } } table.eli5-weights td, .eli5-bg { color: #000000 !important; } </style>"
                    html_expl = responsive_css + html_expl
                    components.html(html_expl, height=450, scrolling=True)
                except KeyError:
                    st.warning("Model explanation is unavailable.")

with tab_bert:
    if model_bert is None or tokenizer is None:
        st.warning("DistilBERT model not found. Run main.py first.")
    else:
        st.success("Using DistilBERT model")
        user_input_bert = st.text_area("Enter a review here (in English):", height=150, key="bert_text")
        if st.button("Analyze Text with DistilBERT"):
            if user_input_bert.strip() == "":
                st.warning("Please enter a valid text.")
            else:
                inputs = tokenizer(
                    user_input_bert, 
                    max_length=128, 
                    padding='max_length', 
                    truncation=True, 
                    return_tensors='pt'
                ).to(device)
                
                with torch.no_grad():
                    outputs = model_bert(**inputs)
                    logits = outputs.logits
                    probs = F.softmax(logits, dim=1)
                    
                    prob_neg = probs[0][0].item()
                    prob_pos = probs[0][1].item()
                    
                    prediction_bert = int(torch.argmax(logits, dim=-1).item())
                
                st.subheader("Result:")
                if prediction_bert == 1:
                    st.success(f"That text is POSITIVE! (Confidence: {prob_pos:.2%})")
                else:
                    st.error(f"That text is NEGATIVE! (Confidence: {prob_neg:.2%})")