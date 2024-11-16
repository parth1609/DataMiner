import streamlit as st
import requests
import os
from typing import List, Tuple
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Directly access the environment variable for Replit
GEMINI_API_KEY = st.secrets['GEMINI_API_KEY']


def extract_text_from_pdf(file) -> Tuple[str, str]:
  """Extract text from a PDF file."""
  try:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
      text += page.extract_text()
    if not text.strip():
      return "", "The PDF appears to be empty or unreadable. It might be scanned or image-based."
    return text, ""
  except Exception as e:
    return "", f"Error extracting text from PDF: {str(e)}"


def extract_text_from_webpage(url: str) -> Tuple[str, str]:
  """Extract text content from a webpage."""
  try:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    text = ' '.join([p.get_text() for p in soup.find_all('p')])
    if not text.strip():
      return "", "No readable text content found on the webpage."
    return text, ""
  except requests.RequestException as e:
    return "", f"Error fetching webpage: {str(e)}"
  except Exception as e:
    return "", f"Error extracting text from webpage: {str(e)}"


def split_text_into_chunks(text: str) -> List[str]:
  """Split text into smaller chunks."""
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000,
      chunk_overlap=200,
      length_function=len,
  )
  return text_splitter.split_text(text)


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
      "x-goog-api-key": GEMINI_API_KEY
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

  # User selects the input type
  input_type = st.radio("Select input type:",
                        ["Local PDF", "PDF URL", "Webpage"])

  all_text = ""
  error_messages = []

  if input_type == "Local PDF":
    uploaded_file = st.file_uploader("Choose a PDF file",
                                     type="pdf",
                                     accept_multiple_files=True)
    for upload_file in uploaded_file:
      if upload_file:
        text, error = extract_text_from_pdf(upload_file)
        all_text += text
        if error:
          error_messages.append(error)

  elif input_type == "PDF URL":
    pdf_urls = st.text_area("Enter PDF URLs (one per line):")
    for pdf_url in pdf_urls.split('\n'):
      if pdf_url.strip():
        try:
          response = requests.get(pdf_url.strip(), timeout=10)
          response.raise_for_status()
          pdf_file = BytesIO(response.content)
          text, error = extract_text_from_pdf(pdf_file)
          all_text += text
          if error:
            error_messages.append(error)
        except requests.RequestException as e:
          error_messages.append(
              f"Error downloading the PDF from {pdf_url}: {str(e)}")
  else:  # Webpage
    webpage_urls = st.text_area("Enter webpage URLs (one per line):")
    for webpage_url in webpage_urls.split('\n'):
      if webpage_url.strip():
        text, error = extract_text_from_webpage(webpage_url.strip())
        all_text += text
        if error:
          error_messages.append(error)

  if error_messages:
    st.error("\n".join(error_messages))

  if all_text:
    chunks = split_text_into_chunks(text)

    question = st.text_area("Ask a question about the content:")
    if question:
      with st.spinner("Generating answer..."):
        relevant_chunks = get_most_relevant_chunks(chunks, question)
        context = " ".join(relevant_chunks)

        answer = get_gemini_response(question, context)

        st.subheader("Answer:")
        st.write(answer)


if __name__ == "__main__":
  main()
