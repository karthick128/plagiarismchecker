import re
from collections import deque
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import heapq

#--------------------TOKENISATION-------------------------

def tokenize(text, n):
    tokens = []
    words = re.findall(r'\w+', text.lower())  # Extract words using regex
    queue = deque(maxlen=n)  # Use deque to keep track of n-grams
    
    for word in words:
        queue.append(word)
        if len(queue) == n:
            tokens.append(' '.join(queue))
    
    return tokens

def process_file(filename, n):
    with open(filename, 'r') as file:
        text = file.read()
        tokens = tokenize(text, n)
        return tokens

# Example usage
n = 3  # Set n-gram size

tokens1 = process_file('file1.txt', n)
tokens2 = process_file('file2.txt', n)

# print("Tokens from file 1:")
# print(tokens1)
# print()
# print("Tokens from file 2:")
# print(tokens2)

#--------------------REMOVAL OF STOPWORDS-------------------------

# Get the list of stopwords
stop_words = set(stopwords.words('english'))

# Remove stop words from the n-gram tokenized words
filtered_ngrams1 = []
filtered_ngrams2 = []

for ngram in tokens1:
    tokenized_ngram = word_tokenize(ngram)
    filtered_ngram = [word for word in tokenized_ngram if word.lower() not in stop_words]
    filtered_ngrams1.append(' '.join(filtered_ngram))

for ngram in tokens2:
    tokenized_ngram = word_tokenize(ngram)
    filtered_ngram = [word for word in tokenized_ngram if word.lower() not in stop_words]
    filtered_ngrams2.append(' '.join(filtered_ngram))


# # Print the filtered n-grams
# print("File1:", filtered_ngrams1)
# print()
# print("File2:", filtered_ngrams2)

# #--------------------KEYWORD EXTRACTION-------------------------

tokenized_words = [ filtered_ngrams1, filtered_ngrams2]

# Convert tokenized words back into documents
documents = [' '.join(tokens) for tokens in tokenized_words]

# Initialize the TF-IDF vectorizer with trigram range
vectorizer = TfidfVectorizer(ngram_range=(1, 3))

# Compute the TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(documents)
# print(tfidf_matrix.shape)

# Get the feature names (trigrams)
feature_names = vectorizer.get_feature_names_out()

# Print the TF-IDF matrix
# for i, document in enumerate(documents):
#     print("Document", i+1)
#     for j, feature_index in enumerate(tfidf_matrix[i].indices):
#         feature_name = feature_names[feature_index]
#         tfidf_score = tfidf_matrix[i, feature_index]
#         print(f"  {feature_name}: {tfidf_score:.4f}")

# #--------------------COSINE SIMILARITY-------------------------

# Compute the cosine similarity between the keyword vectors
similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])

# Convert the similarity score to a plagiarism score
plagiarism_score = similarity_score * 100

# Print the plagiarism score
plagiarism_score = float(plagiarism_score)
print(f"Plagiarism score: {plagiarism_score:.2f}%")



# --------------------GENERATE REPORT WITH HUFFMAN CODING-------------------------

# Get the indices of common trigrams with non-zero TF-IDF scores
common_trigram_indices = np.where(tfidf_matrix[0].toarray() * tfidf_matrix[1].toarray() > 0)

# Get the common trigrams
common_trigrams = [feature_names[idx] for idx in common_trigram_indices[1]]

# Construct a frequency dictionary for the common trigrams
frequency_dict = {}
for trigram in common_trigrams:
    frequency_dict[trigram] = frequency_dict.get(trigram, 0) + 1

# Generate Huffman codes
heap = [[weight, [symbol, ""]] for symbol, weight in frequency_dict.items()]
heapq.heapify(heap)
while len(heap) > 1:
    lo = heapq.heappop(heap)
    hi = heapq.heappop(heap)
    for pair in lo[1:]:
        pair[1] = '0' + pair[1]
    for pair in hi[1:]:
        pair[1] = '1' + pair[1]
    heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

huffman_codes = sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

# Highlight the common text in the original files using Huffman codes
common_text_file1 = []
common_text_file2 = []

with open('file1.txt', 'r') as file1, open('file2.txt', 'r') as file2:
    lines_file1 = file1.readlines()
    lines_file2 = file2.readlines()

    for line in lines_file1:
        for trigram, code in huffman_codes:
            if trigram in line:
                line = line.replace(trigram, f'**{code}**{trigram}**')
        common_text_file1.append(line)

    for line in lines_file2:
        for trigram, code in huffman_codes:
            if trigram in line:
                line = line.replace(trigram, f'**{code}**{trigram}**')
        common_text_file2.append(line)

# Generate the report text
report_text = f"Plagiarism score: {plagiarism_score:.2f}%\n\n"
report_text += "Common text in file 1:\n" + ''.join(common_text_file1) + "\n\n"
report_text += "Common text in file 2:\n" + ''.join(common_text_file2)

# Remove Huffman codes from the report text
report_text = re.sub(r"\*\*\d+\*\*", "", report_text)

# Write the report text to a file
with open('plagiarism_report.txt', 'w') as report_file:
    report_file.write(report_text)

print("Plagiarism report generated successfully.")
