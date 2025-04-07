import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os
import requests
import base64

# ---------------------------
# Background Image
# ---------------------------
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        .stApp::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            height: 100%;
            width: 100%;
            background: rgba(255, 255, 255, 0.85);
            z-index: 0;
        }}
        .block-container {{
            position: relative;
            z-index: 1;
        }}
        .header-title {{
            font-family: 'Arial', sans-serif;
            font-size: 3em;
            color: #800000;
            text-align: center;
        }}
        .scrollable-chat {{
            max-height: 400px;
            overflow-y: auto;
            padding-right: 10px;
        }}
        .chat-message {{
            font-family: 'Courier New', monospace;
            padding: 10px;
            border-radius: 5px;
            background-color: rgba(255, 255, 255, 0.8);
            margin: 5px 0;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("assest/DSC_4712_PRINT.jpg")

# ---------------------------
# App Header
# ---------------------------
st.markdown("<h1 class='header-title'>Gompie the Goat - Your Campus Chatbot</h1>", unsafe_allow_html=True)

# ---------------------------
# Groq API Config
# ---------------------------
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

# ---------------------------
# Load FAISS and Mapping
# ---------------------------
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
INDEX_PATH = "Optimized_Scraped_Data/wpi_corpus_index.faiss"
MAPPING_PATH = "Optimized_Scraped_Data/wpi_corpus_mapping.json"
index = faiss.read_index(INDEX_PATH)
with open(MAPPING_PATH, "r", encoding="utf-8") as f:
    mapping = json.load(f)

# ---------------------------
# Retrieval + Prompt
# ---------------------------
def retrieve_top_k(query, k=3):
    query_embedding = semantic_model.encode([query])
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), k)
    return [mapping[i] for i in indices[0] if i < len(mapping)]

def format_prompt(query, context):
    return f"""You are WPIBot — an expert assistant built for Worcester Polytechnic Institute (WPI) students. 
Use only the information provided in the context below to answer the question accurately and concisely. 
If the answer is not present in the context, respond with \"I couldn't find that information.\"

Context:
{context}

Question: {query}
Answer:"""

# ---------------------------
# Ask Groq
# ---------------------------
def ask_groq(query, context_chunks):
    prompt = format_prompt(query, "\n".join(context_chunks))
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "You are a highly accurate AI assistant, limited to answering based on the provided context."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 300,
        "temperature": 0.6
    }
    response = requests.post(GROQ_API_URL, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"[Groq Error {response.status_code}]"

# ---------------------------
# Streamlit Interface
# ---------------------------
st.write("Ask me anything about WPI!")

chat_container = st.container()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

def send_message():
    user_input = st.session_state.user_input
    if user_input:
        context_chunks = retrieve_top_k(user_input, k=3)
        answer = ask_groq(user_input, context_chunks)
        st.session_state["messages"].append({"role": "You", "content": user_input})
        st.session_state["messages"].append({"role": "Groq (LLaMA3)", "content": answer, "context": context_chunks})
        st.session_state.user_input = ""

# Show Chat
with chat_container:
    st.markdown('<div class="scrollable-chat">', unsafe_allow_html=True)
    for message in st.session_state["messages"]:
        st.markdown(f'<div class="chat-message"><strong>{message["role"]}:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        if "context" in message:
            with st.expander("View Retrieved Context"):
                for i, chunk in enumerate(message["context"]):
                    st.markdown(f"**Chunk {i+1}:** {chunk}")
    st.markdown('</div>', unsafe_allow_html=True)

st.text_input("Type your question here:", key="user_input")
st.button("Send", on_click=send_message)
