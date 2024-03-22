import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, pipeline
from nltk.metrics import edit_distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats
from collections import Counter
import math


def levenshtein_similarity(a, b):
    """Calculate Levenshtein similarity between two strings"""
    return 1 - (edit_distance(a, b) / max(len(a), len(b)))

def jaccard_similarity(a, b):
    """Calculate Jaccard similarity between two sets of tokens"""
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    intersection = a_tokens.intersection(b_tokens)
    union = a_tokens.union(b_tokens)
    return len(intersection) / len(union)

def kl_divergence(a, b):
    """Calculate KL divergence between two probability distributions"""
    a_counts = pd.Series(a).value_counts(normalize=True)
    b_counts = pd.Series(b).value_counts(normalize=True)
    a_probs = a_counts / a_counts.sum()
    b_probs = b_counts / b_counts.sum()
    return scipy.stats.entropy(a_probs, b_probs)


def euclidean_distance(a, b):
    vector1 = Counter(a)
    vector2 = Counter(b)

    # Combine all unique words from both sets
    all_words = set(vector1) | set(vector2)

    # Compute Euclidean distance
    distance = math.sqrt(sum((vector1[word] - vector2[word])**2 for word in all_words))
    return distance


tokenizer = AutoTokenizer.from_pretrained("")
llm_generator = pipeline('text-generation', model='nomic-ai/gpt4all-falcon', tokenizer = tokenizer, trust_remote_code=True)
metadata_df = pd.read_csv('metadata.csv')


def generate_prompt(text):
    return f"Generate an essay in the same style:\n\n{text}"


# Initialize empty lists to store results
results = []

# Main evaluation loop
for _, row in metadata_df.iterrows():
    file_name = row['file_name']
    original_text = row['original-text']
    original_length = row['length']

    prompt = generate_prompt(original_text)

    # Generate text using GPT-3
    generated_text = llm_generator(prompt, max_length=original_length)[0]['generated_text']

    # Calculate evaluation metrics
    levenshtein_sim = levenshtein_similarity(original_text, generated_text)
    jaccard_sim = jaccard_similarity(original_text, generated_text)
    vectorizer = TfidfVectorizer()
    original_vectors = vectorizer.fit_transform([original_text])
    generated_vectors = vectorizer.transform([generated_text])
    cosine_sim = cosine_similarity(original_vectors, generated_vectors)[0][0]
    kl_div = kl_divergence(original_text.split(), generated_text.split())

    # Store results
    results.append({
        'file_name': file_name,
        'original_length': original_length,
        'model': 'gpt3',
        'generated_text': generated_text,
        'generated_length': len(generated_text.split()),
        'levenshtein_similarity': levenshtein_sim,
        'jaccard_similarity': jaccard_sim,
        'cosine_similarity': cosine_sim,
        'kl_divergence': kl_div
    })

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Save the results to a CSV file
results_df.to_csv('evaluation_results.csv', index=False)

print('Evaluation results saved to evaluation_results.csv')
