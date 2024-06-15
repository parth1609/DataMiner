import streamlit as st
import requests
import os
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings)


# Function to extract text from a webpage with error handling
def extract_text_from_webpage(url):
    try:
        loader = AsyncHtmlLoader(url)
        docs = loader.load()  # Assuming load is synchronous and returns a list
        if isinstance(docs, list):
            html2text = Html2TextTransformer()
            docs_transformed = html2text.transform_documents(docs)
            page_info = docs_transformed[0].page_content
            return page_info
        else:
            return ""
    except Exception as e:
        return f"Error: Failed to extract text from webpage ({e})"


# Function to split text into chunks
def split_in_chunks(pagetxt):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200,
                                                   chunk_overlap=20,
                                                   length_function=len,
                                                   is_separator_regex=False)
    texts = text_splitter.create_documents([pagetxt])
    chunks = [text.page_content for text in texts]
    return chunks


def get_gemini_response(question, context):
    """
    Sends a request to the Gemini API and returns the response.
    """
    url = "https://asia-south1-aiplatform.googleapis.com/v1/projects/Generative Language Client/locations/asia-south1/publishers/google/models/gemini-1.5-pro:streamGenerateContent"

    headers = {
        "Authorization": "Bearer " + os.environ['gemini_api'],
        "Content-Type": "application/json"
    }

    data = {
        "prompt":
        f"Context: {context}\n\nQuestion: {question}\n\nAnswer:",  # Format prompt for Gemini
        "temperature": 0.2,  # You can adjust this value for creativity (0-1)
        "maxOutputTokens":
        200  # Adjust the maximum number of tokens in the response
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()[
            "text"]  # Get the generated text from the response
    else:
        return f"Error: Gemini API request failed with status code {response.status_code}"


def split_chunks_in_chroma(chunks, question):
    embedding_function = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2")
    # load it into Chroma
    db = Chroma.from_texts(chunks, embedding_function)  # Use from_texts
    results = db.similarity_search(question)
    return results


# Streamlit app
st.title("LangChain and Gemini Example")

# Input URL
url = st.text_input("Enter the URL of the webpage:")

# Extract text from the webpage
pagetxt = extract_text_from_webpage(url)

# Check if text extraction was successful
if isinstance(pagetxt, str) and pagetxt.startswith("Error"):
    st.error(pagetxt)
else:
    # Split text into chunks
    chunks = split_in_chunks(pagetxt)

    # Create a Streamlit sidebar for user input
    st.sidebar.title("User Input")
    question = st.sidebar.text_input("Enter your question:")

    # Display the extracted text
    st.subheader("Extracted Text")
    st.write(pagetxt)

    # Display the answer
    if chunks and question:
        # Find the most relevant chunks using Chroma
        relevant_chunks = split_chunks_in_chroma(chunks, question)

        # Get the page content from the top result
        if relevant_chunks:
            context = relevant_chunks[0]  # Access page_content

            # Get the response from Gemini
            answer = get_gemini_response(question, context)
            if answer.startswith("Error"):
                st.error(answer)
            else:
                st.subheader("Answer")
                st.write(answer)
        else:
            st.write("No relevant content found.")
