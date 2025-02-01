import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain.document_loaders import PyPDFLoader
import tempfile

# Load environment variables
load_dotenv()
groq_api_key = os.environ['GROQ_API_KEY']

# Set up Streamlit title
st.title("Chatgroq With Llama3 Demo")

# Initialize Groq LLM with API key
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}
""")

# Input field for user question
prompt1 = st.text_input("Enter Your Question From Documents")

# Function to generate vector embeddings and store them in FAISS
def vector_embedding(pdf_file):
    try:
        # Check if embeddings and vectors are already initialized
        if "vectors" not in st.session_state:
            # Initialize embeddings using Ollama
            st.session_state.embeddings = OllamaEmbeddings(model="llama3.2:1b")

            # Save the uploaded PDF file temporarily to disk
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(pdf_file.read())
                temp_file_path = temp_file.name

            # Load PDF content using PyPDFLoader from the temporary file path
            loader = PyPDFLoader(temp_file_path)
            st.session_state.docs = loader.load()

            # Check if documents were loaded
            if not st.session_state.docs:
                st.error("❌ No documents found in the PDF!")
                return

            # Split the text into chunks for better processing
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

            # Check if text splitting works
            if not st.session_state.final_documents:
                st.error("❌ Text splitting failed!")
                return

            # Store the vector embeddings in FAISS
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

            st.success("✅ Vector embeddings created successfully!")
        else:
            st.write("Vector Store DB is already ready!")
    except Exception as e:
        st.error(f"Error during embedding process: {str(e)}")

# File uploader for the user to upload their PDF
pdf_file = st.file_uploader("Upload Your PDF", type=["pdf"])

# Create the vector embeddings when the user uploads a PDF
if pdf_file:
    if st.button("Create Vector Embeddings"):
        vector_embedding(pdf_file)

# Handle the user's query
if prompt1:
    try:
        # Ensure that the vectors are available before querying
        if "vectors" not in st.session_state:
            st.error("❌ Vector Store is not initialized! Please upload a PDF and create embeddings.")
        else:
            # Create the document retrieval chain
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            # Measure response time using wall clock time
            start = time.time()
            response = retrieval_chain.invoke({'input': prompt1})
            st.write("Response time:", round(time.time() - start, 2), "seconds")

            # Display the response
            st.write(response['answer'])

            # Show the document context in an expandable section
            with st.expander("Document Similarity Search"):
                for i, doc in enumerate(response["context"]):
                    st.write(doc.page_content)
                    st.write("--------------------------------")
    except Exception as e:
        st.error(f"Error during query response: {str(e)}")
