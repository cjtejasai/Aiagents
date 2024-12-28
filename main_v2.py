import os
import gc
import tempfile
import uuid
import pandas as pd
import streamlit as st
import logging

from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain.text_splitter import MarkdownTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_community.callbacks.manager import get_openai_callback
from pydantic import BaseModel

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Streamlit App Setup
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME")
OPENAI_API_TYPE = os.getenv("OPENAI_API_TYPE", "azure")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "2023-07-01-preview")


# LLM and Embeddings
@st.cache_resource
def load_llm():
    logging.info("Loading LLM...")
    llm = AzureChatOpenAI(
        openai_api_version=OPENAI_API_VERSION,
        deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
        openai_api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_type=OPENAI_API_TYPE,
        temperature=0,
        streaming=True,
    )
    logging.info("LLM loaded successfully.")
    return llm


@st.cache_resource
def load_embeddings():
    logging.info("Loading embeddings model...")
    embeddings = AzureOpenAIEmbeddings(
        deployment=AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME,
        openai_api_version=OPENAI_API_VERSION,
        openai_api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_type=OPENAI_API_TYPE,
    )
    logging.info("Embeddings model loaded successfully.")
    return embeddings


def reset_chat():
    logging.info("Resetting chat state.")
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()
    logging.info("Chat state reset.")


def display_excel(file):
    logging.info(f"Displaying Excel preview: {file.name}")
    st.markdown("### Excel Preview")
    df = pd.read_excel(file)
    st.dataframe(df)
    logging.info("Excel preview displayed.")


with st.sidebar:
    st.header(f"Add your documents!")
    uploaded_file = st.file_uploader("Choose your `.xlsx` file", type=["xlsx", "xls"])

    if uploaded_file:
        try:
            logging.info(f"File uploaded: {uploaded_file.name}")
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                file_key = f"{session_id}-{uploaded_file.name}"
                st.write("Indexing your document...")
                logging.info(f"File saved to: {file_path}, Key: {file_key}")

                if file_key not in st.session_state.get('file_cache', {}):
                    logging.info("File not in cache, beginning indexing...")
                    loader = UnstructuredExcelLoader(file_path)
                    raw_docs = loader.load()
                    logging.info(f"Loaded raw documents: {len(raw_docs)}.")

                    # Split documents
                    text_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
                    docs = text_splitter.split_documents(raw_docs)
                    logging.info(f"Split documents into {len(docs)} chunks.")

                    # Load embeddings & Create vector store
                    embeddings = load_embeddings()
                    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
                    logging.info(f"Created vector store with {len(docs)} documents.")

                    # Initialize language model and QA chain
                    llm = load_llm()

                    # ====== Customise prompt template ======
                    qa_prompt_tmpl_str = (
                        "Context information is below.\n"
                        "---------------------\n"
                        "{context}\n"
                        "---------------------\n"
                        "Given the context information above I want you to think step by step to answer the query in a highly precise and crisp manner focused on the final answer, incase case you don't know the answer say 'I don't know!'.\n"
                        "Query: {question}\n"
                        "Answer: "
                    )
                    qa_prompt_tmpl = PromptTemplate(template=qa_prompt_tmpl_str,
                                                    input_variables=["context", "question"])
                    logging.info("Custom prompt template created.")

                    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                                           chain_type="stuff",
                                                           retriever=vectorstore.as_retriever(),
                                                           chain_type_kwargs={'prompt': qa_prompt_tmpl},
                                                           return_source_documents=False
                                                           )
                    logging.info("Created RetrievalQA chain.")

                    st.session_state.file_cache[file_key] = qa_chain
                    logging.info(f"QA chain saved to cache for key: {file_key}")

                else:
                    logging.info(f"File found in cache, retrieving cached QA chain for key: {file_key}")
                    qa_chain = st.session_state.file_cache[file_key]

                st.success("Ready to Chat!")
                display_excel(uploaded_file)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            logging.error(f"An error occurred: {e}", exc_info=True)  # Log error with traceback
            st.stop()

col1, col2 = st.columns([6, 1])

with col1:
    st.header(f"RAG over Excel using Langchain & Azure OpenAI")

with col2:
    st.button("Clear ↺", on_click=reset_chat)

if "messages" not in st.session_state:
    reset_chat()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What's up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        qa_chain = None
        for file_key, cached_chain in st.session_state.file_cache.items():
            qa_chain = cached_chain
            break

        if qa_chain:
            logging.info(f"Querying QA chain for prompt: {prompt}")
            with get_openai_callback() as cb:
                response = qa_chain({"query": prompt})
                full_response = response["result"]
            logging.info(f"Response generated: {full_response}")

            message_placeholder.markdown(full_response)
        else:
            logging.warning("No QA chain found, displaying default message.")
            full_response = "Please upload a file to begin chatting!"
            message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})