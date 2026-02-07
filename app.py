import streamlit as st
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


# ==============================
# CONFIG
# ==============================
DB_PATH = "chroma_db"


# ==============================
# LOAD EMBEDDINGS
# ==============================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==============================
# LOAD VECTOR DB
# ==============================
db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)

# ==============================
# LLM
# ==============================
llm = ChatOpenAI(model="gpt-4o-mini")


# ==============================
# FUSION QUERY GENERATOR
# ==============================
def generate_queries(question):
    return [
        question,
        f"Explain {question}",
        f"MSME scheme for {question}",
        f"Government support for {question}",
        f"Eligibility for {question}"
    ]


# ==============================
# RETRIEVAL
# ==============================
def fusion_search(question):
    queries = generate_queries(question)
    docs = []

    for q in queries:
        results = db.similarity_search(q, k=3)
        docs.extend(results)

    # Remove duplicates
    unique_docs = list({d.page_content: d for d in docs}.values())
    return unique_docs


# ==============================
# ANSWER
# ==============================
def get_answer(question):
    docs = fusion_search(question)

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
Answer using only the context below.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


# ==============================
# STREAMLIT UI
# ==============================
st.title("📚 Fusion RAG – MSME Documents")

question = st.text_input("Ask a question from the PDFs")

if st.button("Search"):
    if question:
        with st.spinner("Thinking..."):
            answer = get_answer(question)
        st.write(answer)
