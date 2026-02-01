import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

# --- CONFIGURATION ---
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

if not os.path.exists("vectorstore"):
    os.makedirs("vectorstore")

llm = ChatOllama(model="llama3.2", temperature=0.2, num_predict=1024)
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

CUSTOM_PROMPT_TEMPLATE = """
You are an expert medical assistant. Provide a detailed and comprehensive answer based ONLY on the provided context. 

Guidelines:
- Explain the concepts clearly and in detail.
- If the context contains multiple points, list them using bullet points.
- If the answer is not in the context, say "The provided documents do not specify this information."
- Maintain a professional, helpful medical tone.

Context: {context}
Question: {question}

Detailed Answer:
"""


def get_dynamic_chunk_params(num_pages):
    if num_pages < 5:
        return 500, 50
    elif num_pages < 20:
        return 800, 80
    else:
        return 1200, 120


# --- STREAMLIT UI ---
st.set_page_config(page_title="Medical AI Assistant", layout="wide")
st.title(" PDF Assistant (Llama 3.2)")

with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload your Medical PDF", type="pdf")
    process_btn = st.button("Process & Index")

# --- PROCESSING ---
if uploaded_file and process_btn:
    with st.spinner("Analyzing PDF..."):
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = PyPDFLoader(temp_path)
        documents = loader.load()
        num_pages = len(documents)

        c_size, c_overlap = get_dynamic_chunk_params(num_pages)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=c_size, chunk_overlap=c_overlap)
        text_chunks = text_splitter.split_documents(documents)

        # The metadata (page numbers) is automatically preserved here
        vector_db = FAISS.from_documents(text_chunks, embedding_model)
        vector_db.save_local(DB_FAISS_PATH)

        os.remove(temp_path)
        st.success(f"Indexed {num_pages} pages successfully!")

# --- CHAT ---
st.divider()
query = st.chat_input("Ask a question...")

if query:
    if os.path.exists(DB_FAISS_PATH):
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

        # 1. Search for top 3 relevant chunks
        docs = db.similarity_search(query, k=3)

        # 2. Extract context AND page numbers
        context = "\n\n".join([doc.page_content for doc in docs])

        # Get unique page numbers and add 1 (since index starts at 0)
        pages = sorted(list(set([doc.metadata.get("page", 0) + 1 for doc in docs])))

        # 3. Generate Answer
        prompt = PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"]).format(
            context=context, question=query
        )

        with st.chat_message("assistant"):
            response = llm.invoke(prompt)
            st.write(response.content)

            # Display Sources clearly
            st.caption(f"**Sources:** Found in page(s): {', '.join(map(str, pages))}")
    else:
        st.error("Please upload a PDF first!")