
import faiss
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama  # Faster, local connector
from langchain_core.prompts import PromptTemplate

# --- MODEL INITIALIZATION ---
# We are replacing TinyLlama with Phi-4.
# Make sure you have run 'ollama run phi4' in your terminal first.
llm = ChatOllama(model="llama3.2", temperature=0.2, num_predict=1024)

CUSTOM_PROMPT_TEMPLATE = """
You are a highly factual, document-based assistant.

Rules you MUST follow strictly:
- Only use information explicitly provided in the context.
- Do NOT add, infer, or speculate beyond the context.
- Never provide medical advice or claim cures.
- Do NOT make assumptions or provide general knowledge.
- If the context does not contain a clear answer, reply exactly:
  "The provided documents do not specify this information."
- Keep your answer concise and factual, ideally 1–2 sentences.

Context:
{context}

Question:
{question}

Answer:
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

query = input("Write Query Here: ")

# 1. Similarity search directly from FAISS
docs = db.similarity_search(query, k=3)

# 2. Build context manually
context = "\n\n".join([doc.page_content for doc in docs])

# 3. Create prompt
prompt = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE).format(
    context=context,
    question=query
)

# 4. Call LLM directly (Using Phi-4 now)
llm = model
answer = llm.invoke(prompt)

print(answer.content)