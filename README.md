# 🐐 Gompei the Goat - Your Campus Chatbot
 
 A personalized, RAG-powered chatbot that answers questions related to Worcester Polytechnic Institute (WPI) by retrieving real context and generating intelligent responses using the **Groq LLaMA3 API**.
 
 ![WPI Banner](assest/DSC_4712_PRINT.jpg)
 
 ---
 
 ## 🚀 Features
 
 - 🔎 Context-aware search using FAISS vector store
 - 🧠 Real-time semantic retrieval using Sentence-BERT embeddings
 - 🤖 Natural-sounding answers powered by Groq's blazing-fast LLaMA 3 (70B)
 - 💬 Responsive, scrollable Streamlit interface
 - 🎓 Campus-themed look with WPI’s signature red
 - 🔧 Easy to deploy on EC2 or Streamlit Cloud
 
 ---
 
 ## 🧱 Architecture Overview
 
 ```mermaid
 graph TD
     A[🕸️ Web Crawler] -->|Scrapes WPI content| B[🧠 Sentence-BERT Embeddings]
     B --> C[📦 FAISS Index]
     D[👤 User Query] --> E[🔍 Retrieve top K Chunks from FAISS]
     E --> F[🧾 Format Prompt with Context]
     F --> G[⚡ Groq API - LLaMA3]
     G --> H[💬 Answer in Streamlit App]
