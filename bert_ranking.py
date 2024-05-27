import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load BioBERT model and tokenizer
model_name = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def rank_distractors_bert(question, correct_answer, distractors):
    # Embed the correct answer
    correct_embedding = embed_text(f"{question} {correct_answer}")

    similarities = []
    for distractor in distractors:
        distractor_embedding = embed_text(f"{question} {distractor}")
        similarity = cosine_similarity(correct_embedding, distractor_embedding)
        similarities.append((distractor, similarity[0][0]))

    # Sort distractors by similarity in descending order
    ranked_distractors = sorted(similarities, key=lambda x: x[1], reverse=True)
    
    return [distractor for distractor, sim in ranked_distractors[:3]]

