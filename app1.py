# import streamlit as st
# import os
# from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
# from langchain_community.document_loaders import WebBaseLoader
# from langchain.embeddings import OllamaEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# import time

# from dotenv import load_dotenv
# load_dotenv()

# ## load the Groq API key
# os.environ['NVIDIA_API_KEY']=os.getenv("NVIDIA_API_KEY")

# def vector_embedding():

#     if "vectors" not in st.session_state:

#         st.session_state.embeddings=NVIDIAEmbeddings()
#         st.session_state.loader=PyPDFDirectoryLoader("./us_census") ## Data Ingestion
#         st.session_state.docs=st.session_state.loader.load() ## Document Loading
#         st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=700,chunk_overlap=50) ## Chunk Creation
#         st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:30]) #splitting
#         print("hEllo")
#         st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector OpenAI embeddings


# st.title("Nvidia NIM Demo")
# llm = ChatNVIDIA(model="meta/llama3-70b-instruct")


# prompt=ChatPromptTemplate.from_template(
# """
# Answer the questions based on the provided context only.
# Please provide the most accurate response based on the question
# <context>
# {context}
# <context>
# Questions:{input}

# """
# )


# prompt1=st.text_input("Enter Your Question From Doduments")


# if st.button("Documents Embedding"):
#     vector_embedding()
#     st.write("Vector Store DB Is Ready")

# import time



# if prompt1:
#     document_chain=create_stuff_documents_chain(llm,prompt)
#     retriever=st.session_state.vectors.as_retriever()
#     retrieval_chain=create_retrieval_chain(retriever,document_chain)
#     start=time.process_time()
#     response=retrieval_chain.invoke({'input':prompt1})
#     print("Response time :",time.process_time()-start)
#     st.write(response['answer'])

#     # With a streamlit expander
#     with st.expander("Document Similarity Search"):
#         # Find the relevant chunks
#         for i, doc in enumerate(response["context"]):
#             st.write(doc.page_content)
#             st.write("--------------------------------")
import os
import streamlit as st
import time
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA  # Import RetrievalQA chain

# Load environment variables
load_dotenv()

# Set up the NVIDIA API key
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")

# Define the vector embedding function before it's called
def vector_embedding():
    # Load the document loader and NVIDIA embeddings
    st.session_state.embeddings = NVIDIAEmbeddings()
    st.session_state.loader = PyPDFDirectoryLoader("./us_census")  # Adjust the directory path
    st.session_state.docs = st.session_state.loader.load()  # Load documents

    # Split the documents into chunks
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])

    # Create FAISS vector store from document chunks
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
    st.success("Vector store ready.")

# Page setup
st.set_page_config(page_title="Nvidia NIM Demo", page_icon="🤖")

# Title Section
st.title("Nvidia NIM Demo")
st.subheader("Embedding Documents and Q&A with NVIDIA AI")

# Sidebar for embedding and summarization options
with st.sidebar:
    st.write("### Configure")
    if "vectors" not in st.session_state:
        if st.button("Embed Documents"):
            st.write("Embedding in progress...")
            with st.spinner("Please wait..."):
                vector_embedding()  # Call the embedding function
                st.success("Documents embedded successfully!")

# LLM Setup
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

# Prompt Template for Q&A
prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Input section for the user's question
prompt1 = st.text_input("Enter Your Question From Documents:")

# Q&A Section
if prompt1 and "vectors" in st.session_state:
    with st.spinner("Retrieving and generating the best answer..."):
        # Create document chain
        document_chain = create_stuff_documents_chain(llm, prompt_template)
        
        # Use the retriever from the vector store
        retriever = st.session_state.vectors.as_retriever()
        
        # Create RetrievalQA chain to combine retriever and LLM
        retrieval_qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type="stuff"
        )
        
        # Time the response
        start_time = time.process_time()
        response = retrieval_qa_chain({"query": prompt1})
        response_time = time.process_time() - start_time
        
        # Display the response
        st.write("Response:")
        st.success(response['result'])
        
        # Show the response time
        st.write(f"Response Time: {response_time:.2f} seconds")
        
        # Display retrieved documents with an expander
        with st.expander("Document Similarity Search"):
            st.write("Relevant Document Chunks:")
            for i, doc in enumerate(response["source_documents"]):
                st.write(f"**Document {i + 1}:**")
                st.write(doc.page_content)
                st.write("-" * 50)
else:
    st.warning("Please embed documents before asking questions.")

# Footer
st.write("---")
st.write("Powered by NVIDIA AI and LangChain")




