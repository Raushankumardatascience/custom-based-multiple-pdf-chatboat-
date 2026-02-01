# custom-based-multiple-pdf-chatboat-

# 📄 Medical PDF Chatbot (Llama 3.2 + FAISS + Streamlit)

An AI-powered **Medical PDF Assistant** that allows users to upload medical PDFs and ask questions based **only on the document content**.  
The system uses **Retrieval-Augmented Generation (RAG)** with **Llama 3.2**, **FAISS**, and **HuggingFace embeddings** to provide accurate, context-aware answers along with **source page references**.

---

## 🚀 Features

- 📤 Upload medical PDF documents
- 🧠 Automatic document chunking based on PDF size
- 🔍 Semantic search using FAISS vector database
- 🤖 Context-aware answers using **Llama 3.2 (Ollama)**
- 📄 Displays source **page numbers** for transparency
- 🩺 Professional medical response tone
- ⚡ Fast local inference (no cloud dependency)

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit  
- **LLM**: Llama 3.2 (via Ollama)  
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`  
- **Vector Store**: FAISS  
- **Framework**: LangChain  
- **Language**: Python  

---

├── app.py
├── vectorstore/
│ └── db_faiss/
├── requirements.txt
└── README.md


---

 Installation & Setup

 1️⃣ Clone the Repository
bash
git clone https://github.com/your-username/medical-pdf-chatbot.git
cd medical-pdf-chatbot
2️⃣ Create Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Install & Run Ollama
Download Ollama from: https://ollama.com
Then pull the Llama model:

ollama pull llama3.2
5️⃣ Run the Application
streamlit run app.py


## 📂 Project Structure

