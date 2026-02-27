import os
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage

# -------------------------
# LOAD ENV
# -------------------------
load_dotenv()

# -------------------------
# STREAMLIT UI
# -------------------------
st.set_page_config(page_title="GreenLake Assist", page_icon="🤖")
st.title("💬 GreenLake AI Assistant")

# -------------------------
# FILE TO LOAD
# -------------------------
DATA_FILE = "sample.txt"

# -------------------------
# AUTO RELOAD CHECK
# -------------------------
def get_file_timestamp():
    return os.path.getmtime(DATA_FILE)

# -------------------------
# BUILD VECTOR DB
# -------------------------
@st.cache_resource
def build_vectorstore(file_timestamp):

    # Load document
    loader = TextLoader(DATA_FILE)
    docs = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80
    )
    split_docs = splitter.split_documents(docs)

    # Local embeddings (NO API)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create FAISS index
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    return vectorstore


# rebuild automatically if file updated
timestamp = get_file_timestamp()
vectorstore = build_vectorstore(timestamp)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

st.success("✅ Knowledge Base Ready")

# -------------------------
# LOAD GROQ MODEL
# -------------------------
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

# -------------------------
# CHAT HISTORY
# -------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------
# USER INPUT
# -------------------------
query = st.chat_input("Ask your question...")

if query:

    st.chat_message("user").markdown(query)
    st.session_state.chat_history.append(
        {"role": "user", "content": query}
    )

    # Retrieve context from vector DB
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])

    # Strict RAG Prompt
    prompt = f"""
You are an internal support assistant.

RULES:
- Answer ONLY from context
- If answer not found → say "I don't know"
- Do not guess
- Give step-by-step instructions if procedure

CONTEXT:
{context}

QUESTION:
{query}
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    answer = response.content

    st.chat_message("assistant").markdown(answer)
    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer}
    )
