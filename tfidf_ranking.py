from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def rank_distractors_tfidf(question, correct_answer, distractors):
    # Combine the correct answer and distractors into a single list
    texts = [f"{question} {correct_answer}"] + [f"{question} {distractor}" for distractor in distractors]

    # Compute TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Compute cosine similarity between the correct answer and each distractor
    correct_answer_vector = tfidf_matrix[0]  # The vector for the correct answer
    distractor_vectors = tfidf_matrix[1:]  # The vectors for the distractors

    similarities = cosine_similarity(correct_answer_vector, distractor_vectors)[0]

    # Pair each distractor with its similarity score
    distractor_similarity_pairs = list(zip(distractors, similarities))

    # Sort the distractors by similarity in descending order
    ranked_distractors = sorted(distractor_similarity_pairs, key=lambda x: x[1], reverse=True)

    # Select the top 3 distractors
    top_distractors = [distractor for distractor, similarity in ranked_distractors[:3]]
    
    return top_distractors

