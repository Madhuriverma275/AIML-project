import streamlit as st
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv(r"C:\Users\dell\Downloads\New folder\job_title_des.csv")

# Rename columns if needed
df = df.rename(columns={
    'Job Title': 'job_title',
    'Job Description': 'job_description'
})

# Drop missing values
df = df.dropna()

# Combine text
df['text'] = df['job_title'] + " " + df['job_description']

# Preprocess function
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

df['clean_text'] = df['text'].apply(preprocess)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(df['clean_text'])

# Recommendation function
def recommend_jobs(user_input):
    user_input = preprocess(user_input)
    user_vector = vectorizer.transform([user_input])
    
    similarity = cosine_similarity(user_vector, X)
    scores = similarity[0]
    
    indices = scores.argsort()[::-1]
    
    results = df.iloc[indices][['job_title']].head(5)
    results['match_score'] = scores[indices][:5]
    
    return results

# ---------------- UI ----------------

st.title("💼 AI Job Recommendation System")

st.write("Enter your skills to get job recommendations")

user_input = st.text_area("Enter your skills:")

if st.button("Recommend Jobs"):
    if user_input.strip() != "":
        results = recommend_jobs(user_input)
        
        st.subheader("Top Job Recommendations:")
        
        for i, row in results.iterrows():
            st.write(f"🔹 {row['job_title']} (Score: {row['match_score']:.2f})")
    else:
        st.warning("Please enter some skills")