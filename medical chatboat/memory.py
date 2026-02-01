#pip install transformers accelerate bitsandbytes sentencepiece
#pip install pymupdf
#pip install streamlit
#ollama pull llama3.2

from langchain_community.document_loaders import PyPDFLoader

file_path = "D:\medical_chatboat\Data\chat.pdf"
loader = PyPDFLoader(file_path)

documents = loader.load()

print(len(documents))

from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
                                                 chunk_overlap=50)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=create_chunks(extracted_data=documents)
print("Length of Text Chunks: ", len(text_chunks))


import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings



def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model=get_embedding_model()

# Step 4: Store embeddings in FAISS
DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)
