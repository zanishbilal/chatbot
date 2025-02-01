import streamlit as st
from document_processing import vector_embedding
from query_handling import handle_query

def run_gui():
    # Set up Streamlit title
    st.title("Chatgroq with Llama3 Demo")

    # File uploader for the user to upload their PDF
    pdf_file = st.file_uploader("Upload Your PDF", type=["pdf"])

    # Create the vector embeddings when the user uploads a PDF
    if pdf_file:
        if st.button("Create Vector Embeddings"):
            vector_embedding(pdf_file)

    # Input field for user question
    prompt1 = st.text_input("Enter Your Question From Documents")

    # Handle query if prompt1 is provided
    if prompt1:
        # Call the query handling function to process the query
        handle_query(prompt1)

if __name__ == "__main__":
    run_gui()
