import requests
import random
import nltk
import os
from dotenv import load_dotenv
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv('api_key')

# Download NLTK resources (uncomment if not downloaded already)
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('punkt')

# Function to extract key terms from input text
def get_key_terms(input_text, max_keywords=5):
    lemmatizer = WordNetLemmatizer()

    # Tokenize and lemmatize the input text
    tokens = word_tokenize(input_text.lower())
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Perform part-of-speech tagging
    pos_tags = nltk.pos_tag(lemmatized_tokens)

    # Extract nouns and proper nouns as key terms
    key_terms = [word for word, pos in pos_tags if pos.startswith('N')]

    # Remove stopwords from key terms
    stop_words = set(stopwords.words('english'))
    key_terms = [word for word in key_terms if word not in stop_words]
    key_terms = list(set(key_terms))
    # Shuffle and select a subset of key terms
    random.shuffle(key_terms)
    keywords = key_terms[:max_keywords]
    
    return keywords


def get_related_entities(keywords, api_key):
    base_url = "https://uts-ws.nlm.nih.gov/rest"
    search_url = f"{base_url}/search/current"
    content_url = f"{base_url}/content/current"
    
    # Function to search for CUI of the keyword
    def search_cui(keyword):
        params = {
            'apiKey': api_key,
            'string': keyword,
            'searchType': 'exact'
        }
        response = requests.get(search_url, params=params).json()
        for result in response.get('result', {}).get('results', []):
            if result['name'].lower() == keyword.lower():
                return result['ui']
        return None
    
    # Function to get parent concepts for the CUI
    def get_parent_concepts(cui):
        params = {
            'apiKey': api_key,
            'includeRelationLabels': 'CHD',
            'sabs': 'MTH,RCD,SNMI'
        }
        response = requests.get(f"{content_url}/CUI/{cui}/relations", params=params).json()
        parent_concepts = []
        for result in response.get('result', []):
            if result['relationLabel'] == 'CHD':
                parent_concepts.append(result['relatedIdName'])
        return parent_concepts
    
    # Function to get child concepts for the CUI
    def get_child_concepts(cui):
        params = {
            'apiKey': api_key,
            'includeRelationLabels': 'PAR',
            'sabs': 'SNMI,RCD,MTH',
            'pageNumber': 1
        }
        child_concepts = []
    
        while True:
            response = requests.get(f"{content_url}/CUI/{cui}/relations", params=params).json()
            for result in response.get('result', []):
                if result['relationLabel'] == 'PAR':
                    child_concepts.append(result['relatedIdName'])
            
            # Check if there are more pages
            if params['pageNumber'] >= response.get('pageCount', 0):
                break  # Exit the loop if we've processed all pages
            
            params['pageNumber'] += 1  # Move to the next page
    
        return child_concepts
    
    related_entities = set()
    
    # Get related entities of keywords
    for keyword in keywords:
        cui = search_cui(keyword)
        if not cui:
            continue
        
        # Get parent concepts for the keyword
        parent_concepts = get_parent_concepts(cui)
        for parent_concept in parent_concepts:
            parent_cui = search_cui(parent_concept)
            if not parent_cui:
                continue
            
            # Get child concepts for each parent concept
            child_concepts = get_child_concepts(parent_cui)
            for child_concept in child_concepts:
                related_entities.add(child_concept.lower())
    
    # Remove keywords from related entities
    related_entities.difference_update(map(str.lower, keywords))
    
    return list(related_entities)
