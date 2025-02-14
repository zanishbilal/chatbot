# import tempfile
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.embeddings import OllamaEmbeddings
# import streamlit as st
# from langchain_community.vectorstores import Pinecone
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from pinecone import Pinecone, ServerlessSpec
# import os
# from dotenv import load_dotenv
# from langchain_openai import OpenAIEmbeddings

# load_dotenv()



# PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
# PINECONE_INDEX = os.getenv('PINECONE_INDEX')
# # GPT_API_KEY = os.getenv('GPT_API_KEY')

# os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

# # Initialize Pinecone client
# pc = Pinecone(api_key=PINECONE_API_KEY)


# index_name = PINECONE_INDEX

  
 

# def vector_embedding(pdf_file):
#     try:
#         # Check if embeddings and vectors are already initialized
#         if "vectors" not in st.session_state:
#             # Initialize embeddings using Ollama
            
            
#             # embaddings=st.session_state.embeddings = OllamaEmbeddings(model="llama3.2:1b")
#             embaddings=st.session_state.embeddings = OpenAIEmbeddings()
            

#             # Save the uploaded PDF file temporarily to disk
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
#                 temp_file.write(pdf_file.read())
#                 temp_file_path = temp_file.name

#             # Load PDF content using PyPDFLoader from the temporary file path
#             loader = PyPDFLoader(temp_file_path)
#             st.session_state.docs = loader.load()

#             # Check if documents were loaded
#             if not st.session_state.docs:
#                 st.error("❌ No documents found in the PDF!")
#                 return

#             # Split the text into chunks for better processing
#             st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=15000, chunk_overlap=8000)
#             final_documents = st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

#     # Check if text splitting works
            
#             # Split the documents into chunks

#             # Check if text splitting works
#             if not st.session_state.final_documents:
#                 st.error("❌ Text splitting failed!")
#                 return

#             # Generate embeddings for the chunks
#             chunk_embeddings = embaddings.embed_documents([doc.page_content for doc in st.session_state.final_documents])
#             print("Chunk embeddings", len(chunk_embeddings))

#             # Ensure embeddings match the number of chunks
#             if len(chunk_embeddings) != len(st.session_state.final_documents):
#                 st.error(f"Embedding count mismatch: {len(chunk_embeddings)} vs {len(st.session_state.final_documents)}")
            
#             # Create a dictionary for chunk embeddings (including chunk content)
#             chunk_embeddings_dict = {}
#             for idx, (doc, embedding) in enumerate(zip(st.session_state.final_documents, chunk_embeddings)):
#                 doc_id = doc.metadata.get("id", f"chunk_{idx}")  # Using chunk index as fallback for unique ID
#                 chunk_embeddings_dict[doc_id] = {
#                     "embedding": embedding,
#                     "metadata": {
#                         **doc.metadata,  # Include the existing metadata
#                         "content": doc.page_content  # Storing the chunk's content in the metadata
#                     }
#                 }

#             # Check if the index exists, and create it if it doesn't
#             if index_name not in pc.list_indexes().names():
#                 pc.create_index(
#                     name=index_name, 
#                     dimension=1536,  # Correct dimension for llama3.2:1b model (4096)
#                     metric="cosine",  # Adjust the metric if needed
#                     spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Using AWS us-east-1 as the region
#                 )

#             # Insert the chunk embeddings into Pinecone
#             vectors_to_insert = []
#             for doc_id, data in chunk_embeddings_dict.items():
#                 vectors_to_insert.append({
#                     "id": doc_id,  # Unique ID for the chunk
#                     "values": [float(value) for value in data["embedding"]],  # Ensure embedding values are floats
#                     "metadata": data["metadata"]  # Metadata for the chunk, including content
#                 })

#             # Use the correct method to insert into Pinecone
#             index = pc.Index(index_name)  # Initialize the index with the string name
#             index.upsert(vectors=vectors_to_insert, namespace="ns1")

#             st.success("✅ Chunk vector embeddings inserted into Pinecone successfully!")

#             # Store the vectors in session state for retrieval later
#             st.session_state.vectors = index

#     except Exception as e:
#         st.error(f"Error during embedding process: {str(e)}")



import tempfile
import fitz  # PyMuPDF
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
import streamlit as st
from langchain_community.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX = os.getenv('PINECONE_INDEX')
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = PINECONE_INDEX

def extract_equations_from_pdf(pdf_file):
    equations = []
    try:
        # Open the PDF using PyMuPDF
        doc = fitz.open(pdf_file)
        for page in doc:
            text = page.get_text("text")  # Extract text
            equations.extend(extract_latex_from_text(text))  # Function to extract equations from the text
        return equations
    except Exception as e:
        print(f"Error extracting equations: {str(e)}")
        return []

def extract_latex_from_text(text):
    """ Function to detect LaTeX expressions from plain text (requires logic to detect LaTeX code) """
    latex_equations = []
    # This can be customized based on how equations appear in your PDF text
    # Look for patterns that represent equations (e.g., math symbols, $$, \[...\], etc.)
    lines = text.split('\n')
    for line in lines:
        if "$$" in line or "\\[" in line or "\\]" in line:
            latex_equations.append(line)
    return latex_equations

def vector_embedding(pdf_file):
    try:
        # Check if embeddings and vectors are already initialized
        if "vectors" not in st.session_state:
            # Initialize embeddings using OpenAI (or Ollama if preferred)
            embaddings = st.session_state.embeddings = OpenAIEmbeddings()

            # Save the uploaded PDF file temporarily to disk
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(pdf_file.read())
                temp_file_path = temp_file.name

            # Extract equations from the PDF
            equations = extract_equations_from_pdf(temp_file_path)
            print(f"Equations found: {len(equations)}")

            # Load PDF content using PyPDFLoader
            loader = PyPDFLoader(temp_file_path)
            st.session_state.docs = loader.load()

            # Check if documents were loaded
            if not st.session_state.docs:
                st.error("❌ No documents found in the PDF!")
                return

            # Extract equations from the documents
            st.session_state.docs_with_equations = []
            for doc in st.session_state.docs:
                content = doc.page_content
                doc_with_eq = {
                    "content": content,
                    "equations": equations  # Add detected equations to document
                }
                st.session_state.docs_with_equations.append(doc_with_eq)

            # Split the text into chunks for better processing
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=15000, chunk_overlap=8000)
            final_documents = st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

            # Check if text splitting works
            if not st.session_state.final_documents:
                st.error("❌ Text splitting failed!")
                return

            # Generate embeddings for the chunks, including equations
            chunk_embeddings = embaddings.embed_documents([doc.page_content for doc in st.session_state.final_documents])
            print("Chunk embeddings", len(chunk_embeddings))

            # Ensure embeddings match the number of chunks
            if len(chunk_embeddings) != len(st.session_state.final_documents):
                st.error(f"Embedding count mismatch: {len(chunk_embeddings)} vs {len(st.session_state.final_documents)}")

            # Create a dictionary for chunk embeddings (including chunk content and equations)
            chunk_embeddings_dict = {}
            for idx, (doc, embedding) in enumerate(zip(st.session_state.final_documents, chunk_embeddings)):
                doc_id = doc.metadata.get("id", f"chunk_{idx}")  # Using chunk index as fallback for unique ID
                chunk_embeddings_dict[doc_id] = {
                    "embedding": embedding,
                    "metadata": {
                        **doc.metadata,  # Include the existing metadata
                        "content": doc.page_content,  # Storing the chunk's content in the metadata
                        "equations": st.session_state.docs_with_equations[idx]["equations"]  # Add equations to metadata
                    }
                }

            # Check if the index exists, and create it if it doesn't
            if index_name not in pc.list_indexes().names():
                pc.create_index(
                    name=index_name, 
                    dimension=1536,  # Correct dimension for OpenAI model (4096 for LLaMA model)
                    metric="cosine",  # Adjust the metric if needed
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Using AWS us-east-1 as the region
                )

            # Insert the chunk embeddings into Pinecone
            vectors_to_insert = []
            for doc_id, data in chunk_embeddings_dict.items():
                vectors_to_insert.append({
                    "id": doc_id,  # Unique ID for the chunk
                    "values": [float(value) for value in data["embedding"]],  # Ensure embedding values are floats
                    "metadata": data["metadata"]  # Metadata for the chunk, including content and equations
                })

            # Use the correct method to insert into Pinecone
            index = pc.Index(index_name)  # Initialize the index with the string name
            index.upsert(vectors=vectors_to_insert, namespace="ns1")

            st.success("✅ Chunk vector embeddings inserted into Pinecone successfully!")

            # Store the vectors in session state for retrieval later
            st.session_state.vectors = index

    except Exception as e:
        st.error(f"Error during embedding process: {str(e)}")
