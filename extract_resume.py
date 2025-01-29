import fitz  # PyMuPDF
import re

def extract_full_text_from_pdf(pdf_path):
    """
    Extracts full text from a PDF without any truncation.
    :param pdf_path: Path to the PDF file
    :return: Full extracted text from the PDF
    """
    doc = fitz.open(pdf_path)
    full_text = "\n".join(page.get_text("text") for page in doc)
    return full_text


def clean_resume_text(resume_text):
    """
    Preprocess and optimize extracted resume text.
    :param resume_text: Raw text extracted from a PDF
    :return: Cleaned and formatted resume text
    """
    # Remove multiple spaces and special symbols
    resume_text = re.sub(r'\s+', ' ', resume_text)  # Replace multiple spaces/newlines with single space
    resume_text = re.sub(r'[^a-zA-Z0-9,.@|\s-]', '', resume_text)  # Remove non-ASCII characters
    resume_text = resume_text.strip()
    return resume_text


# Example usage: Provide the PDF path dynamically (to be handled in Flask app)
pdf_path = None  # This will be dynamically set in Flask
resume_text = extract_full_text_from_pdf(pdf_path) if pdf_path else None
cleaned_resume = clean_resume_text(resume_text) if resume_text else None
