from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
from werkzeug.utils import secure_filename
import json
from sklearn.metrics.pairwise import cosine_similarity

# Import project modules
from extract_resume import extract_full_text_from_pdf, clean_resume_text
from parse_resume import parse_resume, get_resume_feedback
from embed_resume import create_resume_embedding, load_glove_embeddings

# Initialize Flask App
app = Flask(__name__)

# Upload folder configuration
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load Job Data & GloVe Embeddings
#JOB_DATA_FILE = "uploads/job_embedding.csv"  # Job dataset (already embedded)
GLOVE_FILE = "uploads/glove.6B.100d.txt"  # GloVe file path

print("Loading job data...")
df = pd.read_pickle('uploads/job_embedding.pkl')
#df["glove_embedding"] = df["glove_embedding"].apply(lambda x: np.array(eval(x)))  # Convert to NumPy array

print("Loading GloVe embeddings...")
glove_embeddings = load_glove_embeddings(GLOVE_FILE)

print("Setup complete!")

@app.route("/get_countries", methods=["GET"])
def get_countries():
    """Returns a sorted list of unique countries from job data."""
    countries = sorted(df["country"].unique().tolist())  # Get unique country list from df
    return jsonify({"countries": countries})


@app.route("/")
def index():
    """Render home page for resume upload."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_resume():
    """
    Handles resume PDF upload, extracts data, computes embedding, 
    performs cosine similarity, and returns job recommendations.
    """
    if "resume" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["resume"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Securely save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Extract & clean text
        resume_text = extract_full_text_from_pdf(file_path)
        cleaned_resume = clean_resume_text(resume_text)

        # Parse structured resume + feedback
        structured_resume = parse_resume(cleaned_resume)
        resume_feedback = get_resume_feedback(resume_text)

        # Debug: Print structured resume & feedback in the terminal
        print("\n🔍 STRUCTURED RESUME (Extracted from Mistral AI):\n", json.dumps(structured_resume, indent=4))
        print("\n💡 RESUME FEEDBACK:\n", resume_feedback)

        # Create resume embedding
        resume_embedding = create_resume_embedding(structured_resume, glove_embeddings, embedding_dim=100)

        # Compute cosine similarity with job dataset
        job_embeddings = np.stack(df["glove_embedding"].values)
        similarities = cosine_similarity([resume_embedding], job_embeddings)[0]
        df["similarity_score"] = similarities

        # Sort and return top matching jobs
        df_sorted = df.sort_values(by="similarity_score", ascending=False)
        
        # Filter by country selection (from user input)
        selected_countries = request.form.getlist("countries")  # Get selected countries from UI
        if selected_countries:
            df_sorted = df_sorted[df_sorted["country"].isin(selected_countries)]

        # Ensure job titles are unique
        df_unique_jobs = df_sorted.drop_duplicates(subset=["job title"], keep="first")

        # Select top N unique jobs
        num_jobs = int(request.form.get("num_jobs", 5))
        job_recommendations = df_unique_jobs.head(num_jobs)[["job title", "company", "location", "country", "work type"]].to_dict(orient="records")


        # Convert NumPy arrays to Python lists before returning JSON
        # Remove 'glove_embedding' key from recommendations since it's not needed in frontend
        for job in job_recommendations:
            job.pop("glove_embedding", None)  # Safely remove if exists

        return jsonify({
            "resume_feedback": resume_feedback,
            "job_recommendations": job_recommendations
        })


    return jsonify({"error": "File processing failed"}), 500


if __name__ == "__main__":
    app.run(debug=True)