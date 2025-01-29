import json
from openai import OpenAI

# Initialize Hugging Face Mistral API using OpenAI-compatible structure
client = OpenAI(
    base_url="",
    api_key=""
)

def parse_resume(cleaned_text):
    """
    Extract structured resume data using Mistral API via OpenAI-compatible endpoint.
    """
    prompt = f"""
    Extract the following details from the provided resume text.
    Extract **only what is available**, and return empty values if missing.
    **Keep experience summary super concise.**

    **Return JSON format only.**

    Extract:
    - "skills": List of skills across resume separated by ','.
    - "experience_summary": Super concise work experience summary.
    - "years_of_experience": Total number of years worked related to experience section.
    - "previous_job_titles": List of past job titles.

    Resume Text:
    {cleaned_text}

    **Output only valid JSON. Example:**
    {{
        "skills": ["list", "of", "all", "technical", "non-technical", "soft", "skills", "across", "resume", "just list them", "separated by comma ','"],
        "experience_summary": "brief overall summary of work experience (combined across different roles, NOT role-specific, NOT company-specific)...",
        "years_of_experience": integer,
        "previous_job_titles": ["list", "of", "previous", "roles"]
    }}
    """

    chat_completion = client.chat.completions.create(
        model="tgi",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )

    # Extract response text
    response = chat_completion.choices[0].message.content

    if not response or "{" not in response:
        raise ValueError("Mistral API returned an empty or invalid response.")

    # Extract JSON from response
    json_start = response.find("{")
    json_end = response.rfind("}") + 1
    json_str = response[json_start:json_end]

    return json.loads(json_str)


def get_resume_feedback(resume_text):
    """
    Generate a concise 50-word feedback for the resume using Mistral API.
    """
    prompt = f"""
    Provide a concise, motivational, and constructive feedback for the following resume.
    Highlight strengths, suggest improvements, and maintain a maximum word limit of 50 words.

    Resume Text:
    {resume_text}

    **Output example:**
    "Your resume highlights impressive skills in Python and data engineering. To improve further, consider adding measurable achievements. Overall, a strong resume showcasing versatility and expertise!"
    """

    chat_completion = client.chat.completions.create(
        model="tgi",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )

    # Extract response text
    response = chat_completion.choices[0].message.content.strip()

    if not response:
        raise ValueError("Mistral API returned an empty or invalid feedback.")

    return response


# Process the cleaned resume text
structured_resume = parse_resume(cleaned_resume)  # Structured data
resume_feedback = get_resume_feedback(resume_text)  # Feedback on resume text
