import joblib
import fitz  # PyMuPDF
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDIA_DIR = os.path.join(BASE_DIR, 'media')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'final_fake_job_classifier.pkl')
model = joblib.load(MODEL_PATH)

def extract_text_from_pdf_path(filename):
    pdf_path = os.path.join(MEDIA_DIR, filename)
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")
    
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def parse_text_to_features(text):
    # Create a dictionary with all fields your model expects
    # You should split/extract real parts if possible. Here we put all text in 'description' as example
    return {
        'title': '',  # Extract if possible or empty string
        'description': text,
        'company_profile': '',
        'requirements': '',
        'required_experience': 0,
        'required_education': 0
    }

def predict_from_pdf(filename):
    text = extract_text_from_pdf_path(filename)
    if not text.strip():
        raise ValueError("The PDF contains no extractable text.")
    
    features_dict = parse_text_to_features(text)
    
    # Convert dict to DataFrame (1 row)
    input_df = pd.DataFrame([features_dict])
    
    # Predict
    prediction = model.predict(input_df)
    return prediction[0] 
