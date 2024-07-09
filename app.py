import streamlit as st
import requests
import os
from typing import List
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_text_from_webpage(url: str) -> str:
  """Extract text content from a webpage."""
  try:
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return ' '.join([p.get_text() for p in soup.find_all('p')])
  except Exception as e:
    st.error(f"Error extracting text from webpage: {e}")
    return ""


def split_text_into_chunks(text: str, chunk_size: int = 500) -> List[str]:
  """Split text into smaller chunks."""
  words = text.split()
  return [
      ' '.join(words[i:i + chunk_size])
      for i in range(0, len(words), chunk_size)
  ]


def get_most_relevant_chunks(chunks: List[str],
                             question: str,
                             top_k: int = 2) -> List[str]:
  """Get the most relevant chunks for a given question using TF-IDF and cosine similarity."""
  vectorizer = TfidfVectorizer()
  tfidf_matrix = vectorizer.fit_transform(chunks + [question])

  chunk_similarities = cosine_similarity(tfidf_matrix[-1],
                                         tfidf_matrix[:-1])[0]
  top_chunk_indices = chunk_similarities.argsort()[-top_k:][::-1]

  return [chunks[i] for i in top_chunk_indices]


def get_gemini_response(question: str, context: str) -> str:
  """Send a request to the Gemini API and return the response."""
  url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"

  headers = {
      "Content-Type": "application/json",
      "x-goog-api-key": os.environ['GEMINI_API_KEY']
  }

  data = {
      "contents": [{
          "parts": [{
              "text":
              f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
          }]
      }],
      "generationConfig": {
          "temperature": 0.2,
          "topK": 40,
          "topP": 0.95,
          "maxOutputTokens": 1024,
      }
  }

  try:
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["candidates"][0]["content"]["parts"][0]["text"]
  except requests.RequestException as e:
    return f"Error: Failed to get response from Gemini API. {str(e)}"


def main():
  st.title("DataMiner")
  st.subheader("Ask a question and get a response from the web")

  # User input
  url = st.text_input("Enter the URL of the webpage:")

  if url:
    with st.spinner("Processing webpage..."):
      #  Extract text from webpage
      text = extract_text_from_webpage(url)

      if text:
        #  Split text into chunks
        chunks = split_text_into_chunks(text)

        #  User question and answer generation
        question = st.text_input("Ask a question about the webpage content:")
        if question:
          with st.spinner("Generating answer..."):
            # Retrieve relevant chunks
            relevant_chunks = get_most_relevant_chunks(chunks, question)
            context = " ".join(relevant_chunks)

            # Get answer from Gemini API
            answer = get_gemini_response(question, context)

            st.subheader("Answer:")
            st.write(answer)
      else:
        st.error(
            "Failed to extract text from the webpage. Please check the URL and try again."
        )


if __name__ == "__main__":
  main()
