import streamlit as st
import os
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter


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


from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings)


def split_chunks_in_chroma(chunks, question):
    embedding_function = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2")
    # load it into Chroma
    db = Chroma.from_documents(chunks, embedding_function)
    docs = db.similarity_search(question)
    return docs  # Return only the top result


# Streamlit app
st.title("LangChain and Streamlit Example")

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

    # Display the chunks
    if chunks:
        st.subheader("Chunks")
        for i, chunk in enumerate(chunks):
            st.write(f"Chunk {i+1}:")
            st.write(chunk)

    # Display the similarity search result
    if chunks and question:
        st.subheader("Most Relevant Chunk to your Question")
        answer = split_chunks_in_chroma(chunks, question)
        st.write(answer)
