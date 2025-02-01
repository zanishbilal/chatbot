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
from langchain_community.document_loaders import PyPDFLoader
import tempfile
from langchain_community.vectorstores import Pinecone
import pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain.schema import HumanMessage
 # Ensure you have the correct import for ChatGroq
from groq import Groq  # Import Groq client



PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_ENV = os.getenv("PINECONE_ENV")

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "chatbot"  

# Load environment variables
load_dotenv()
groq_api_key = os.environ['GROQ_API_KEY']



# Set up Streamlit title
st.title("Chatgroq With Llama3 Demo")

# Initialize Groq LLM with API key


# Input field for user question
prompt1 = st.text_input("Enter Your Question From Documents")

# Function to generate vector embeddings and store them in FAISS
def vector_embedding(pdf_file):
    try:
        # Check if embeddings and vectors are already initialized
        if "vectors" not in st.session_state:
            # Initialize embeddings using Ollama
            embaddings=st.session_state.embeddings = OllamaEmbeddings(model="llama3.2:1b")

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
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=1000)
            final_documents = st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

    # Check if text splitting works
            
            # Split the documents into chunks

            # Check if text splitting works
            if not st.session_state.final_documents:
                st.error("❌ Text splitting failed!")
                return

            # Generate embeddings for the chunks
            chunk_embeddings = embaddings.embed_documents([doc.page_content for doc in st.session_state.final_documents])
            print("Chunk embeddings", len(chunk_embeddings))

            # Ensure embeddings match the number of chunks
            if len(chunk_embeddings) != len(st.session_state.final_documents):
                st.error(f"Embedding count mismatch: {len(chunk_embeddings)} vs {len(st.session_state.final_documents)}")
            
            # Create a dictionary for chunk embeddings (including chunk content)
            chunk_embeddings_dict = {}
            for idx, (doc, embedding) in enumerate(zip(st.session_state.final_documents, chunk_embeddings)):
                doc_id = doc.metadata.get("id", f"chunk_{idx}")  # Using chunk index as fallback for unique ID
                chunk_embeddings_dict[doc_id] = {
                    "embedding": embedding,
                    "metadata": {
                        **doc.metadata,  # Include the existing metadata
                        "content": doc.page_content  # Storing the chunk's content in the metadata
                    }
                }

            # Check if the index exists, and create it if it doesn't
            if index_name not in pc.list_indexes().names():
                pc.create_index(
                    name=index_name, 
                    dimension=4096,  # Correct dimension for llama3.2:1b model (4096)
                    metric="cosine",  # Adjust the metric if needed
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Using AWS us-east-1 as the region
                )

            # Insert the chunk embeddings into Pinecone
            vectors_to_insert = []
            for doc_id, data in chunk_embeddings_dict.items():
                vectors_to_insert.append({
                    "id": doc_id,  # Unique ID for the chunk
                    "values": data["embedding"],  # Embedding values for the chunk
                    "metadata": data["metadata"]  # Metadata for the chunk, including content
                })

            # Use the correct method to insert into Pinecone
            index = pc.Index(index_name)  # Initialize the index with the string name
            index.upsert(vectors=vectors_to_insert, namespace="ns1")

            st.success("✅ Chunk vector embeddings inserted into Pinecone successfully!")

            # Store the vectors in session state for retrieval later
            st.session_state.vectors = index

    except Exception as e:
        st.error(f"Error during embedding process: {str(e)}")


# File uploader for the user to upload their PDF
pdf_file = st.file_uploader("Upload Your PDF", type=["pdf"])

# Create the vector embeddings when the user uploads a PDF
if pdf_file:
    if st.button("Create Vector Embeddings"):
        vector_embedding(pdf_file)

# Define prompt template


client = Groq()

# Define your prompt template
prompt_template = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
</context>
Questions:{input}
""")

# Handle the query if `prompt1` is provided
if prompt1:
    # Ensure that the vectors are available before querying
    if "vectors" not in st.session_state:
        st.error("❌ Vector Store is not initialized! Please upload a PDF and create embeddings.")
    else:
        # Get the embedding for the query prompt
        query_embedding = st.session_state.embeddings.embed_documents([prompt1])[0]

        # Start measuring the response time
        start = time.process_time()

        # Query Pinecone using the query embedding
        response = st.session_state.vectors.query(
            namespace="ns1",  # Specify the namespace
            vector=query_embedding,  # Using the query embedding for vector search
            top_k=2,  # Number of similar documents to return
            include_values=True,  # Include values (vectors)
            include_metadata=True,  # Include metadata
        )

        print("Response:", response)

        # Display the response time
        st.write("Response time:", round(time.process_time() - start, 2), "seconds")

        # Check if we received any matches in the response
        if 'matches' not in response or len(response['matches']) == 0:
            st.error("❌ No matches found for the query.")
        else:
            # Extract content from the matches for summarization
            content_list = [match['metadata'].get('content', '') for match in response['matches'] if 'metadata' in match and 'content' in match['metadata']]
            content = "\n".join(content_list) if content_list else None

            # Format the prompt using the context and the query input
            formatted_prompt = prompt_template.format(input=prompt1, context=content)

            # Generate the summary using your model
            try:
                completion = client.chat.completions.create(
                    model="Llama3-8b-8192",  # Specify your model (Llama3 here as an example)
                    messages=[
                        {"role": "system", "content": "Summarize the following text:"},
                        {"role": "user", "content": formatted_prompt}
                    ],
                    temperature=1,
                    max_completion_tokens=1024,
                    top_p=1,
                    stream=True,
                    stop=None,
                )

                # Capture and print the summarized text
                summary_text = "".join(chunk.choices[0].delta.content or "" for chunk in completion)

                # Display the final summarized answer
                st.write(summary_text)

            except Exception as e:
                st.error(f"❌ Error in summarizing the response: {str(e)}")

            # Optionally, show more context in an expandable section
            with st.expander("Document Context"):
                for match in response['matches']:
                    st.write(match['metadata'].get('content', 'No content available'))
                    st.write("--------------------------------")