import fitz  # PyMuPDF
import re

def extract_text_from_pdf(pdf_path):
    """Extract full text from a PDF."""
    doc = fitz.open(pdf_path)
    full_text = "\n".join(page.get_text("text") for page in doc)
    return full_text

def clean_resume_text(resume_text):
    """Preprocess and clean extracted resume text."""
    resume_text = re.sub(r'\s+', ' ', resume_text)  # Replace multiple spaces/newlines with single space
    resume_text = re.sub(r'[^a-zA-Z0-9,.@|\s-]', '', resume_text)  # Remove special characters
    return resume_text.strip()
