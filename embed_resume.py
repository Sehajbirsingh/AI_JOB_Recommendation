import numpy as np

# Load GloVe embeddings
def load_glove_embeddings(file_path):
    """
    Loads pre-trained GloVe embeddings from a file into a dictionary.
    :param file_path: Path to GloVe embedding file (e.g., 'glove.6B.100d.txt')
    :return: Dictionary {word: embedding_vector}
    """
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefficients = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefficients
    print(f"Loaded {len(embeddings_index)} word vectors.")
    return embeddings_index


# Function to get the average GloVe vector for a given text
def get_average_glove_vector(text, glove_embeddings, embedding_dim=100):
    """
    Converts text into an averaged GloVe embedding vector.
    :param text: Text to embed
    :param glove_embeddings: Dictionary containing word embeddings
    :param embedding_dim: Dimension of word embeddings
    :return: Numpy array representing the text embedding
    """
    words = text.split()
    word_vectors = [glove_embeddings[word] for word in words if word in glove_embeddings]
    
    if word_vectors:
        return np.mean(word_vectors, axis=0)  # Average all word embeddings
    else:
        return np.zeros(embedding_dim)  # Return a zero vector if no words found


# Function to generate resume embedding
def create_resume_embedding(structured_resume, glove_embeddings, embedding_dim=100):
    """
    Creates a single embedding representation for a resume.
    :param structured_resume: JSON output from Mistral AI (skills, experience summary)
    :param glove_embeddings: Loaded GloVe embedding dictionary
    :param embedding_dim: Dimension of embeddings
    :return: Resume embedding vector
    """
    # Combine skills and experience summary into a single text
    resume_text = " ".join(structured_resume["skills"]) + " " + structured_resume["experience_summary"]
    
    # Generate GloVe embedding for resume text
    resume_embedding = get_average_glove_vector(resume_text, glove_embeddings, embedding_dim)
    
    return resume_embedding


# Load GloVe embeddings (Adjust path if needed)
glove_embeddings = load_glove_embeddings("uploads/glove.6B.100d.txt")

# Generate Resume Embedding
resume_embedding = create_resume_embedding(structured_resume, glove_embeddings, embedding_dim=100)
