# import time
# from langchain_groq import ChatGroq
# from langchain_openai import OpenAIEmbeddings
# from langchain_core.prompts import ChatPromptTemplate
# import pinecone
# from pinecone import Pinecone
# from groq import Groq 
# import streamlit as st
# from dotenv import load_dotenv
# import os
# pc = Pinecone(api_key="pcsk_6aafqg_2rxeEF81794Cg2CAG8d7bNMvfn5RUYUt9SUgGdNGC4DxLjC2saqje9V8PKoookm")
# index_name = "chatbot"
# load_dotenv()
# groq_api_key = os.environ['GROQ_API_KEY']
# client = Groq()  # Groq Client for querying Groq model
# prompt_template = ChatPromptTemplate.from_template("""
# Answer the questions based on the provided context only.
# Please provide the most accurate response based on the question.
# <context>
# {context}
# </context>
# Questions:{input}
# """)
# def handle_query(prompt1):
#     # Ensure that vectors are initialized before querying
#     if "vectors" not in st.session_state:
#         st.error("❌ Vector Store is not initialized! Please upload a PDF and create embeddings.")
#         return
    
#     # Get the embedding for the query prompt
#     query_embedding = st.session_state.embeddings.embed_documents([prompt1])[0]

#     # Start measuring the response time
#     start = time.process_time()

#     # Query Pinecone using the query embedding
#     response = st.session_state.vectors.query(
#         namespace="ns1",  # Specify the namespace
#         vector=query_embedding,  # Using the query embedding for vector search
#         top_k=2,  # Number of similar documents to return
#         include_values=True,  # Include values (vectors)
#         include_metadata=True,  # Include metadata
#     )

#     print("Response:", response)

#     # Display the response time
#     st.write("Response time:", round(time.process_time() - start, 2), "seconds")

#     # Check if we received any matches in the response
#     if 'matches' not in response or len(response['matches']) == 0:
#         st.error("❌ No matches found for the query.")
#         return

#     # Extract content from the matches for summarization
#     content_list = [match['metadata'].get('content', '') for match in response['matches'] if 'metadata' in match and 'content' in match['metadata']]
#     content = "\n".join(content_list) if content_list else None

    
    
#     formatted_prompt = prompt_template.format(input=prompt1, context=content)

#     try:
#         completion = client.chat.completions.create(
#             model="Llama3-8b-8192",  # Specify your model (Llama3 here as an example)
#             messages=[
#                 {"role": "system", "content": "Summarize the following text:"},
#                 {"role": "user", "content": formatted_prompt}
#             ],
#             temperature=1,
#             max_completion_tokens=1024,
#             top_p=1,
#             stream=True,
#             stop=None,
#         )

#         # Capture and print the summarized text
#         summary_text = "".join(chunk.choices[0].delta.content or "" for chunk in completion)

#         # Display the final summarized answer
#         st.write(summary_text)

#     except Exception as e:
#         st.error(f"❌ Error in summarizing the response: {str(e)}")

#     # Optionally, show more context in an expandable section
#     with st.expander("Document Context"):
#         for match in response['matches']:
#             st.write(match['metadata'].get('content', 'No content available'))
#             st.write("--------------------------------")


import time
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
import pinecone
from pinecone import Pinecone
from groq import Groq 
import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.embeddings import OllamaEmbeddings
import time
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from groq import Groq 
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Pinecone instance
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Create an index (if not already created)
index_name = "chatbot"


# Create a prompt template
prompt_template = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
</context>
Questions:{input}
""")

def handle_query(prompt1):
    # Get the embedding for the query prompt using Ollama
    embeddings = st.session_state.embeddings = OllamaEmbeddings(model="llama3.2:1b")
    query_embedding = embeddings.embed_documents([prompt1])[0]

    # Start measuring the response time
    start = time.process_time()

    # Connect to Pinecone index and perform query
    index = pc.Index(index_name)  # Use the Pinecone instance to create an index object
    response = index.query(
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
        st.error("""❌ 1 Vector database is empty. 
                 2 No matches found for the query""")
        return

    # Extract content from the matches for summarization
    content_list = [match['metadata'].get('content', '') for match in response['matches'] if 'metadata' in match and 'content' in match['metadata']]
    content = "\n".join(content_list) if content_list else None

    formatted_prompt = prompt_template.format(input=prompt1, context=content)

    try:
        # Use Groq for summarization
        client = Groq()
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
