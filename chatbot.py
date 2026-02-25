import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# ----------------------------------
# PAGE
# ----------------------------------
st.set_page_config(page_title="GreenLake Assistant")
st.title("HPE GreenLake Support Assistant")

# ----------------------------------
# OPENAI KEY
# ----------------------------------
api_key = st.text_input("Enter OpenAI API Key", type="password")

if not api_key:
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

# ----------------------------------
# LOAD LOCAL DOCUMENT
# ----------------------------------
@st.cache_resource
def load_rag():

    # read document
    with open("knowledge.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    docs = splitter.create_documents([text])

    # embeddings
    embeddings = OpenAIEmbeddings()

    # vector db
    vectorstore = FAISS.from_documents(docs, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # STRICT PROMPT
    template = """
You are an expert HPE GreenLake support assistant.

RULES:
- Answer ONLY from context.
- Do NOT use outside knowledge.
- If answer not in context → say "I don't know".
- Do NOT guess.
- Provide step-by-step answers for procedures.

CONTEXT:
{context}

QUESTION:
{question}

FINAL ANSWER:
"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

    return qa


qa_chain = load_rag()

st.success("✅ Knowledge Base Loaded")

# ----------------------------------
# CHAT
# ----------------------------------
query = st.text_input("Ask your question")

if query:
    answer = qa_chain.run(query)
    st.write("### Answer")
    st.write(answer)
