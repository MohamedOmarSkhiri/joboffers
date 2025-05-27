import joblib
import fitz  # PyMuPDF
import os
import pandas as pd
import re

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

def build_feature_vector_from_pdf_text(pdf_text):
    text = pdf_text.replace('\r', '').replace('\xa0', ' ').strip()

    def extract_section(keyword, text, next_keywords):


        
        pattern = rf'{keyword}:(.*?)(?=' + '|'.join([re.escape(k) + ':' for k in next_keywords]) + '|$)'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    sections = {
        "description": ["JOB RESPONSIBILITIES", "REQUIRED QUALIFICATIONS", "REMUNERATION", "APPLICATION PROCEDURES"],
        "requirements": ["REMUNERATION", "APPLICATION PROCEDURES", "OPENING DATE"],
        "benefits": ["APPLICATION PROCEDURES", "OPENING DATE", "ABOUT COMPANY"]
    }

    description = extract_section("JOB DESCRIPTION", text, sections["description"])
    requirements = extract_section("REQUIRED QUALIFICATIONS", text, sections["requirements"])
    benefits = extract_section("REMUNERATION/SALARY", text, sections["benefits"])
    company_profile = extract_section("ABOUT COMPANY", text, ["$"]) or extract_section("ABOUT COMPANY", text, ["APPLICATION DEADLINE"])

    df = pd.DataFrame([{
        "company_profile": company_profile,
        "description": description,
        "requirements": requirements,
        "benefits": benefits,
    }])

    df["description_length"] = df["description"].apply(lambda x: len(x))
    df["text_combined"] = df["description"] + " " + df["requirements"] + " " + df["benefits"]
    df["has_benefits"] = df["benefits"].apply(lambda x: 1 if len(x.strip()) > 0 else 0)
    df["has_requirements"] = df["requirements"].apply(lambda x: 1 if len(x.strip()) > 0 else 0)

    optional_cols = [
        "employment_type", "required_experience", "required_education",
        "industry", "function", "salary_range", "telecommuting",
        "has_company_logo", "has_questions"
    ]
    for col in optional_cols:
        df[col] = 0

    return df

def predict_from_pdf(filename):
    text = extract_text_from_pdf_path(filename)
    if not text.strip():
        raise ValueError("The PDF contains no extractable text.")
    
    input_df = build_feature_vector_from_pdf_text(text)
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]  # probability of class 1 (Fake)

    return {
        "prediction": "Fake" if prediction == 1 else "Real",
        "probability_fake": round(probability, 4)
    }
