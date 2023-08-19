import re
from collections import deque
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import heapq

from flask import Flask, render_template, request, redirect, url_for

import sqlite3

app = Flask(__name__)

# ----------------------------------------DATABASE------------------------------

def create_users_table():
    conn = sqlite3.connect('Dupchecker.db')
    cursor = conn.cursor()

    # Create users table
    cursor.execute('''CREATE TABLE IF NOT EXISTS users
                      (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      username TEXT UNIQUE NOT NULL,
                      password TEXT NOT NULL,
                      email_id TEXT NOT NULL)''')

    conn.commit()
    conn.close()

def insert_user(username, password, email):

    conn = sqlite3.connect('Dupchecker.db')
    cursor = conn.cursor()

    # Insert user into the table
    cursor.execute("INSERT INTO users (username, password, email_id) VALUES (?, ?, ?)", (username, password, email))

    conn.commit()
    conn.close()

def check_username_exists(username):
    conn = sqlite3.connect('Dupchecker.db')
    cursor = conn.cursor()

    # Execute the SELECT query
    cursor.execute("SELECT COUNT(*) FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()

    # Check if the count is greater than 0
    if result[0] > 0:
        res = 1
    else:
        res = 0

    conn.close()
    return res

def retrieve_pwd(username):
    conn = sqlite3.connect('Dupchecker.db')
    cursor = conn.cursor()

    cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
    pwd = cursor.fetchone()
    
    return pwd[0]

#---------------------------------BACKEND--------------------------------------
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

    #--------------------REMOVAL OF STOPWORDS-------------------------
def remove_stopwords(tokens1,tokens2):
    stop_words = set(stopwords.words('english'))

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
    return filtered_ngrams1, filtered_ngrams2
    #--------------------KEYWORD EXTRACTION-------------------------

def extract_keywords(filtered_ngrams1, filtered_ngrams2):
    tokenized_words = [ filtered_ngrams1, filtered_ngrams2]

    
    documents = [' '.join(tokens) for tokens in tokenized_words]
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    return tfidf_matrix, feature_names

     #--------------------COSINE SIMILARITY-------------------------
def similarity(tfidf_matrix):
    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    plagiarism_score = similarity_score * 100
    plagiarism_score = float(plagiarism_score)  
    return plagiarism_score

# --------------------GENERATE REPORT WITH HUFFMAN CODING-------------------------

def huffman(tfidf_matrix, feature_names, score):
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
    report_text = f"Plagiarism score: {score:.2f}%\n\n"
    report_text += "Common text in file 1:\n" + ''.join(common_text_file1) + "\n\n"
    report_text += "Common text in file 2:\n" + ''.join(common_text_file2)

    # Remove Huffman codes from the report text
    report_text = re.sub(r"\*\*\d+\*\*", "", report_text)

    # Write the report text to a file
    with open('plagiarism_report.txt', 'w') as report_file:
        report_file.write(report_text)

    #-------------------------MAIN-----------------------------------
def main(file1, file2):
    f1 = process_file(file1, 3)
    f2 = process_file(file2, 3)

    fng1, fng2 = remove_stopwords(f1, f2)
    mat, f_names = extract_keywords(fng1, fng2)
    sim = similarity(mat)
    huffman(mat, f_names, sim)
    return sim
#------------------------------------------------SIGNUP------------------------------------

@app.route('/', methods=['GET', 'POST'])
def index1():
    create_users_table()
    if request.method == "POST":
        username = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")
        if check_username_exists(username) == 0:
            insert_user(username, password, email)
            return redirect(url_for("index3"))
        elif check_username_exists(username) == 1:
            return render_template('index1.html', error='Account with the given credentials exists!')   
    return render_template("index1.html")

#------------------------------------------------LOGIN---------------------------------------------

@app.route('/index2', methods=['GET','POST'])
def index2():
    if request.method == 'POST':

        username = request.form.get('username')
        password = request.form.get('password')
        if check_username_exists(username) == 1:
            pwd = retrieve_pwd(username)
            if pwd == password:
                return redirect(url_for('index3'))
            else:
                return render_template('index2.html', error='Invalid credentials!')
        elif check_username_exists(username) == 0:
            return redirect(url_for('index1'))
    return render_template('index2.html')
        
#-------------------------------------------INPUT------------------------------------------------

@app.route('/index3', methods=['GET','POST'])
def index3():
    if request.method == "POST":
        file1 = request.form.get("file1")
        file2 = request.form.get("file2")
        global score 
        score = main(file1, file2)
        return redirect(url_for('index4'))
    return render_template('index3.html')

#--------------------------------------------OUTPUT------------------------------------------------

@app.route('/index4', methods = ["GET"])
def index4():
    return render_template('index4.html', variable = str(round(score,2))+ '%')


if __name__ == '__main__':
    app.run(debug = True)
