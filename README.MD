# Job Recommendation System using Deep Learning

## Overview
This project is a **Job Recommendation System** that uses **Deep Learning** and **Natural Language Processing (NLP)** to recommend the most suitable jobs to users based on their resumes. The system parses resumes to extract key information (e.g., skills, experience, education) and matches them with job postings using **cosine similarity** and **autoencoder embeddings**. Additionally, it provides feedback on how to improve the resume using **Mistral AI**.

---

## Features
1. **Resume Parsing**:
   - Extracts skills, years of experience, certifications, education, and responsibilities using **NLP** and **regex**.
2. **Job Matching**:
   - Uses **TF-IDF** and **autoencoder embeddings** to represent resumes and job postings.
   - Computes **cosine similarity** to recommend the top N jobs.
3. **Feedback System**:
   - Provides actionable feedback using **Mistral AI** to improve the resume (e.g., highlighting responsibilities, adding objectives).

---

## Workflow
1. **Data Collection**:
   - Job postings dataset downloaded from [Kaggle](https://www.kaggle.com/datasets/PromptCloudHQ/us-technology-jobs-on-dicecom).
2. **Data Preprocessing**:
   - Clean and preprocess job postings and resumes.
   - Convert text into numerical representations using **TF-IDF**.
3. **Model Training**:
   - Train an **autoencoder** to generate embeddings for job postings and resumes.
4. **Recommendation System**:
   - Use **cosine similarity** to match resumes with job postings.
5. **Feedback Generation**:
   - Use **Mistral AI** to provide feedback on how to improve the resume.
6. **Web Application**:
   - Deploy the system as a web app for users to upload resumes and receive recommendations.

---

## Technologies Used
- **Python Libraries**:
  - `pandas`, `numpy` for data manipulation.
  - `scikit-learn` for TF-IDF and cosine similarity.
  - `tensorflow` and `keras` for building the autoencoder.
  - `spaCy` and `NLTK` for NLP and resume parsing.
  - `regex` for pattern matching.
- **Mistral AI**:
  - Used for generating feedback on resumes.
- **Web Framework**:
  - `Flask` or `Streamlit` for building the web app.

---

## Code Implementation

### **1. Install Required Libraries**
```bash
pip install pandas numpy scikit-learn tensorflow spacy