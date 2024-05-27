from csg_broad_narrow import get_key_terms, get_related_entities, api_key
from tfidf_ranking import rank_distractors_tfidf
from bert_ranking import rank_distractors_bert


# Sample question and answer
question = "Urinary metabolite of progesterone"
answer = "Pregnanediol"

# Extract keywords from the question
keywords = get_key_terms(question)
# Add the answer as an additional keyword
keywords.append(answer)

print("\nKeywords:", keywords)

candidate_distractors = get_related_entities(keywords, api_key)
print("\nCandidate distractors", candidate_distractors)


top_3_distractors = rank_distractors_tfidf(question, answer, candidate_distractors)
print("\ntfidf Ranked Distractors", top_3_distractors)

top_3_distractors = rank_distractors_bert(question, answer, candidate_distractors)
print("\nbiobert Ranked Distractors", top_3_distractors)
