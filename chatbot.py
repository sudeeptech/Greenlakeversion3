# =====================================================
# GREENLAKE RAG CHATBOT (LOCAL EMBEDDINGS + GROQ)
# =====================================================

import os
from dotenv import load_dotenv
import streamlit as st

from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------------------------
# LOAD ENV VARIABLES (.env file)
# -------------------------
load_dotenv()

# -------------------------
# STREAMLIT UI
# -------------------------
st.set_page_config(page_title="GreenLake Assist", page_icon="🤖")
st.title("💬 GreenLake AI Assist")

# -------------------------
# RELOAD BUTTON (FOR NEW DATA)
# -------------------------
if st.button("🔄 Reload Knowledge Base"):
    st.cache_resource.clear()
    st.rerun()

# -------------------------
# CHAT HISTORY
# -------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------
# RAG SETUP (AUTO RELOAD WHEN FILE CHANGES)
# -------------------------
@st.cache_resource
def setup_rag(file_time):  # file_time forces refresh when document updates

    # Load document
    loader = TextLoader("sample.txt")
    docs = loader.load()

    # Split document into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    split_docs = splitter.split_documents(docs)

    # LOCAL EMBEDDINGS (FREE — NO API)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create vector database
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    return vectorstore.as_retriever(search_kwargs={"k": 3})


# Detect document update automatically
file_time = os.path.getmtime("sample.txt")
retriever = setup_rag(file_time)

# -------------------------
# GROQ MODEL
# -------------------------
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

# -------------------------
# USER INPUT
# -------------------------
user_prompt = st.chat_input("Ask about HPE GreenLake...")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append(
        {"role": "user", "content": user_prompt}
    )

    # Retrieve relevant content
    docs = retriever.invoke(user_prompt)
    context = "\n".join([doc.page_content for doc in docs])

    # Strict RAG prompt
    rag_prompt = f"""
You are an internal company support assistant.

RULES:
- Answer ONLY from provided context
- Do NOT use outside knowledge
- If answer not found → say "I don't know"
- Be clear and step-by-step

CONTEXT:
{context}

QUESTION:
{user_prompt}

FINAL ANSWER:
"""

    response = llm.invoke([HumanMessage(content=rag_prompt)])
    answer = response.content

    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer}
    )

    with st.chat_message("assistant"):
        st.markdown(answer)
